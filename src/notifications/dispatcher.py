"""
notifications/dispatcher.py – Orchestrates policy, drivers, retry, and throttling.

NotificationDispatcher
    The single entry point for the notification subsystem.

    ``dispatch(alert)``
        Evaluates the escalation policy for the alert type, creates a
        ``notifications`` row for each applicable channel, sends to the WebSocket
        driver synchronously (for near-zero latency), and dispatches SMTP/SMS in
        background threads to avoid blocking Flask request handlers or MQTT
        subscriber threads.

    Throttling
        Per ``(channel, alert_type)`` pair: at most one notification per
        ``throttle_s`` seconds (default 60 s; override via ``PCS_NOTIFY_THROTTLE_S``).
        Throttled records are persisted with ``status="throttled"`` so the audit
        trail stays complete.

    Retry
        A daemon thread wakes every ``retry_interval_s`` seconds (default 10 s)
        and re-attempts ``status="failed"`` records that have a past (or absent)
        ``next_retry`` timestamp and fewer than ``max_retries`` total attempts.
        Back-off formula: ``delay = min(2^attempt × 5, 300)`` seconds.

Environment variables
---------------------
PCS_NOTIFY_THROTTLE_S   Minimum seconds between duplicate (channel, type) notifications (default 60).
PCS_NOTIFY_MAX_RETRIES  Maximum delivery attempts before giving up (default 3).
PCS_NOTIFY_RETRY_S      Seconds between retry-worker sweeps (default 10).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from .drivers import NotificationDriver, SmtpNotConfiguredError
from .policy  import EscalationPolicy
from .store   import NotificationStore

log = logging.getLogger(__name__)


class NotificationDispatcher:
    """
    Orchestrates policy resolution, driver dispatch, throttling, and retry.

    Parameters
    ----------
    store:
        :class:`NotificationStore` used to persist notification records.
    policy:
        :class:`EscalationPolicy` that maps alert severity → channel list.
    drivers:
        Iterable of :class:`NotificationDriver` instances.  The dispatcher
        selects drivers by ``driver.name``; unknown channel names are skipped
        with a warning.
    throttle_s:
        Minimum seconds between two notifications on the same
        ``(channel, alert_type)`` pair.  ``0`` disables throttling.
    max_retries:
        Maximum total delivery attempts per notification record
        (including the first attempt).  ``0`` disables retries.
    retry_interval_s:
        Seconds between retry-worker sweeps (default 10).

    Usage
    -----
    ::

        dispatcher = NotificationDispatcher(
            store=notify_store,
            policy=EscalationPolicy(),
            drivers=[WebSocketDriver(socketio), SmtpDriver(), NullSmsDriver()],
        )
        dispatcher.start()   # start retry daemon thread

        # In _on_alert callback:
        dispatcher.dispatch(alert_dict)
    """

    def __init__(
        self,
        store:            NotificationStore,
        policy:           EscalationPolicy,
        drivers:          list[NotificationDriver],
        *,
        throttle_s:       float = float(os.environ.get("PCS_NOTIFY_THROTTLE_S", "60")),
        max_retries:      int   = int(os.environ.get("PCS_NOTIFY_MAX_RETRIES",  "3")),
        retry_interval_s: float = float(os.environ.get("PCS_NOTIFY_RETRY_S",   "10")),
    ) -> None:
        self._store            = store
        self._policy           = policy
        self._drivers: dict[str, NotificationDriver] = {d.name: d for d in drivers}
        self._throttle_s       = throttle_s
        self._max_retries      = max_retries
        self._retry_interval_s = retry_interval_s

        # throttle tracker: (channel, alert_type) → monotonic time of last send
        self._throttle: dict[tuple[str, str], float] = {}
        self._throttle_lock = threading.Lock()

        self._retry_thread: threading.Thread | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def dispatch(self, alert: dict[str, Any], severity: str | None = None) -> list[dict[str, Any]]:
        """
        Evaluate the escalation policy and attempt delivery to all applicable channels.

        The WebSocket driver is invoked **synchronously** so the browser receives
        the alert with minimal latency.  All other drivers (SMTP, SMS) are
        dispatched in **background threads**.

        Parameters
        ----------
        alert:
            The alert / event dict.  Must contain at least ``"alert_type"``.
        severity:
            Explicit severity override.  If *None*, the policy's
            :meth:`EscalationPolicy.get_severity` is queried using ``alert_type``.

        Returns
        -------
        list[dict]
            Notification records created during this dispatch (one per channel,
            including throttled ones).
        """
        alert_type = alert.get("alert_type") or "unknown"
        alert_id   = alert.get("id")

        if severity is None:
            severity = self._policy.get_severity(alert_type)

        channels = self._policy.get_channels(severity)
        created: list[dict[str, Any]] = []

        for channel in channels:
            driver = self._drivers.get(channel)
            if driver is None:
                log.debug("[dispatcher] no driver for channel=%s", channel)
                continue

            # ── Throttle check ────────────────────────────────────────────────
            throttle_key = (channel, alert_type)
            now_s = time.monotonic()
            with self._throttle_lock:
                last_s = self._throttle.get(throttle_key, 0.0)
                if self._throttle_s > 0 and (now_s - last_s) < self._throttle_s:
                    n = self._store.add(
                        alert_id  = alert_id,
                        alert_type= alert_type,
                        severity  = severity,
                        channel   = channel,
                        payload   = alert,
                        status    = "throttled",
                    )
                    log.debug(
                        "[dispatcher] throttled channel=%s alert_type=%s "
                        "(%.1f s < %.1f s window)",
                        channel, alert_type, now_s - last_s, self._throttle_s,
                    )
                    created.append(n)
                    continue

            # ── Create notification record ────────────────────────────────────
            n = self._store.add(
                alert_id  = alert_id,
                alert_type= alert_type,
                severity  = severity,
                channel   = channel,
                payload   = alert,
            )

            # ── Deliver: WebSocket is synchronous; others run in background ───
            if channel == "websocket":
                n = self._attempt_send(n, driver)
                if n["status"] == "sent":
                    with self._throttle_lock:
                        self._throttle[throttle_key] = time.monotonic()
                created.append(n)

            else:
                # Capture loop variable for the thread closure
                _n, _d, _tk = n, driver, throttle_key

                def _bg_send(notif=_n, drv=_d, tk=_tk) -> None:
                    result = self._attempt_send(notif, drv)
                    if result["status"] == "sent":
                        with self._throttle_lock:
                            self._throttle[tk] = time.monotonic()

                t = threading.Thread(target=_bg_send, daemon=True,
                                     name=f"pcs-notify-{channel}")
                t.start()
                created.append(n)  # return the pending record; it will update async

        return created

    def start(self) -> None:
        """Start the background retry-worker daemon thread."""
        if self._max_retries == 0:
            return
        self._retry_thread = threading.Thread(
            target=self._retry_worker,
            daemon=True,
            name="pcs-notify-retry",
        )
        self._retry_thread.start()
        log.info(
            "[dispatcher] retry worker started  interval=%.0fs max_retries=%d",
            self._retry_interval_s, self._max_retries,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _attempt_send(
        self,
        notification: dict[str, Any],
        driver:       NotificationDriver,
    ) -> dict[str, Any]:
        """
        Try a single delivery attempt.

        Updates the notification record in the store and returns the refreshed dict.
        """
        n_id = notification["id"]
        self._store.increment_attempts(n_id)
        attempts_after = (notification.get("attempts") or 0) + 1

        try:
            driver.send(notification)
            self._store.update_status(n_id, "sent")
            log.info(
                "[dispatcher] sent channel=%s alert_type=%s id=%d attempt=%d",
                notification.get("channel"), notification.get("alert_type"),
                n_id, attempts_after,
            )

        except SmtpNotConfiguredError as exc:
            # SMTP is intentionally unconfigured — mark failed but do NOT retry
            self._store.update_status(n_id, "failed", error_msg=str(exc))
            log.warning(
                "[dispatcher] smtp not configured; notification %d skipped: %s",
                n_id, exc,
            )

        except Exception as exc:
            if attempts_after < self._max_retries:
                delay_s    = min(2 ** attempts_after * 5, 300)
                next_retry = (
                    datetime.now(timezone.utc) + timedelta(seconds=delay_s)
                ).isoformat()
                self._store.update_status(
                    n_id, "failed",
                    error_msg  = str(exc),
                    next_retry = next_retry,
                )
                log.warning(
                    "[dispatcher] failed channel=%s id=%d attempt=%d/%d "
                    "next_retry=+%.0fs: %s",
                    notification.get("channel"), n_id,
                    attempts_after, self._max_retries, delay_s, exc,
                )
            else:
                self._store.update_status(n_id, "failed", error_msg=str(exc))
                log.error(
                    "[dispatcher] giving up channel=%s id=%d after %d attempts: %s",
                    notification.get("channel"), n_id, attempts_after, exc,
                )

        refreshed = self._store.get(n_id)
        return refreshed if refreshed is not None else notification

    def _retry_worker(self) -> None:
        """Background daemon: retry failed notifications on schedule."""
        log.debug("[dispatcher-retry] retry worker running")
        while True:
            try:
                time.sleep(self._retry_interval_s)
                pending = self._store.get_pending_retry(max_retries=self._max_retries)
                if pending:
                    log.debug("[dispatcher-retry] %d notification(s) to retry", len(pending))
                for n in pending:
                    driver = self._drivers.get(n.get("channel", ""))
                    if driver is None:
                        log.debug(
                            "[dispatcher-retry] no driver for channel=%s, skipping id=%s",
                            n.get("channel"), n.get("id"),
                        )
                        continue
                    result = self._attempt_send(n, driver)
                    if result.get("status") == "sent":
                        tk = (n.get("channel", ""), n.get("alert_type", ""))
                        with self._throttle_lock:
                            self._throttle[tk] = time.monotonic()

            except Exception as exc:
                log.debug("[dispatcher-retry] unexpected error: %s", exc)
