"""
notifications/drivers.py – Delivery channel drivers.

Each driver implements :class:`NotificationDriver` (a simple ABC with a single
``send(notification)`` method) and carries a ``name`` class attribute that
matches the channel string used in :class:`EscalationPolicy`.

Included drivers
----------------
WebSocketDriver
    Emits a Socket.IO event to all connected dashboard clients.  Instantiated
    with the ``SocketIO`` object so no global state is needed.  The socket event
    name is derived from ``alert_type`` (e.g. ``"safety_alert"`` → ``"alert"``,
    ``"loitering"`` → ``"loitering"``).

SmtpDriver
    Sends a plain-text email via :mod:`smtplib`.  Completely configured through
    environment variables; raises :class:`SmtpNotConfiguredError` (a
    :class:`RuntimeError` subclass) when mandatory env vars are absent so the
    dispatcher can log and skip rather than crash.

SmsDriverBase (ABC)
    Abstract base class for SMS adapters.  Subclass it, implement ``send()``,
    and pass your instance to :class:`NotificationDispatcher`.  No vendor
    credentials are baked in.

NullSmsDriver
    Default no-op adapter: logs the would-be SMS message at DEBUG level and
    returns without calling any external service.  Replace it by providing a
    concrete :class:`SmsDriverBase` subclass.

Environment variables (SmtpDriver)
-----------------------------------
PCS_SMTP_HOST      SMTP server hostname (required to enable SMTP delivery)
PCS_SMTP_PORT      Port number (default: 587)
PCS_SMTP_USER      Login username (optional; skip auth if absent)
PCS_SMTP_PASSWORD  Login password (optional)
PCS_SMTP_FROM      Sender address (default: pcs@localhost)
PCS_SMTP_TO        Comma-separated recipient addresses
PCS_SMTP_TLS       Use STARTTLS: "1" (default) or "0"
PCS_SMTP_START_TLS Alias for PCS_SMTP_TLS (takes precedence if set)
"""

from __future__ import annotations

import logging
import os
import smtplib
import textwrap
from abc import ABC, abstractmethod
from email.mime.text import MIMEText
from typing import Any

log = logging.getLogger(__name__)

# ── Socket event routing ───────────────────────────────────────────────────────
# Maps alert_type → Socket.IO event name.
_WS_EVENT_MAP: dict[str, str] = {
    "forced_entry": "alert",
    "tailgate":     "alert",
    "safety_alert": "alert",
    "loitering":    "loitering",
    "left_object":  "left_object",
}


# ── Base class ─────────────────────────────────────────────────────────────────


class NotificationDriver(ABC):
    """Abstract base class for all notification channel drivers."""

    #: Channel name used in the escalation policy.
    name: str = ""

    @abstractmethod
    def send(self, notification: dict[str, Any]) -> None:
        """
        Deliver a notification.

        Parameters
        ----------
        notification:
            The notification record dict (as returned by :class:`NotificationStore`).
            The ``"payload"`` key contains the original alert / event dict.

        Raises
        ------
        Exception
            Any exception raised here will be caught by the dispatcher, which
            will mark the notification as ``"failed"`` and schedule a retry.
        """

    def is_configured(self) -> bool:
        """Return True if this driver has the credentials / config it needs."""
        return True


# ── WebSocket driver ───────────────────────────────────────────────────────────


class WebSocketDriver(NotificationDriver):
    """
    Delivers notifications by emitting a Socket.IO broadcast event.

    The emitted event name is determined by the ``alert_type`` field in the
    notification payload:

    * ``forced_entry``, ``tailgate``, ``safety_alert``  →  ``"alert"``
    * ``loitering``                                      →  ``"loitering"``
    * ``left_object``                                    →  ``"left_object"``

    Any unknown type defaults to ``"alert"``.

    Parameters
    ----------
    socketio:
        The :class:`flask_socketio.SocketIO` instance created in
        :func:`create_app`.
    """

    name = "websocket"

    def __init__(self, socketio: Any) -> None:
        self._sio = socketio

    def send(self, notification: dict[str, Any]) -> None:
        payload    = notification.get("payload") or {}
        alert_type = payload.get("alert_type") or notification.get("alert_type", "")
        event_name = _WS_EVENT_MAP.get(alert_type, "alert")
        self._sio.emit(event_name, payload)
        log.debug(
            "[ws-driver] emitted event=%s alert_type=%s id=%s",
            event_name, alert_type, payload.get("id"),
        )


# ── SMTP driver ────────────────────────────────────────────────────────────────


class SmtpNotConfiguredError(RuntimeError):
    """Raised when mandatory SMTP environment variables are missing."""


class SmtpDriver(NotificationDriver):
    """
    Delivers notifications by sending a plain-text email via :mod:`smtplib`.

    Configuration is read from environment variables at **send time** so that
    the driver can be instantiated unconditionally and will gracefully skip
    delivery when not configured.

    Required env var (enables SMTP)
    --------------------------------
    ``PCS_SMTP_HOST`` – SMTP server hostname.

    Optional env vars
    -----------------
    See module docstring for the full list.
    """

    name = "smtp"

    def is_configured(self) -> bool:
        return bool(os.environ.get("PCS_SMTP_HOST", "").strip())

    def _build_message(self, notification: dict[str, Any]) -> MIMEText:
        payload    = notification.get("payload") or {}
        alert_type = payload.get("alert_type") or notification.get("alert_type", "unknown")
        severity   = notification.get("severity", "?")
        camera_id  = payload.get("camera_id",  "N/A")
        door_id    = payload.get("door_id",    "N/A")
        confidence = payload.get("confidence", "N/A")
        ts         = payload.get("timestamp_utc") or notification.get("created_at", "")
        meta       = payload.get("meta") or {}

        body = textwrap.dedent(f"""\
            Privacy Campus Security — {alert_type.upper()} [{severity}]

            Timestamp  : {ts}
            Type       : {alert_type}
            Severity   : {severity}
            Camera     : {camera_id}
            Door       : {door_id}
            Confidence : {confidence}
            Meta       : {meta}

            This notification was generated automatically.
            Log in to the dashboard to acknowledge this alert.
        """)

        from_addr = os.environ.get("PCS_SMTP_FROM", "pcs@localhost")
        to_addr   = os.environ.get("PCS_SMTP_TO",   from_addr)
        subject   = (
            f"[PCS/{severity.upper()}] {alert_type.replace('_', ' ').title()} "
            f"— camera {camera_id}"
        )

        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"]    = from_addr
        msg["To"]      = to_addr
        return msg

    def send(self, notification: dict[str, Any]) -> None:
        """
        Send an email notification.

        Raises
        ------
        SmtpNotConfiguredError
            When ``PCS_SMTP_HOST`` is not set; the dispatcher logs this as a
            warning and marks the notification ``"failed"`` without retrying.
        smtplib.SMTPException
            On SMTP protocol / auth errors; triggers the retry scheduler.
        """
        host = os.environ.get("PCS_SMTP_HOST", "").strip()
        if not host:
            raise SmtpNotConfiguredError(
                "PCS_SMTP_HOST is not set; SMTP delivery is disabled."
            )

        port     = int(os.environ.get("PCS_SMTP_PORT", "587"))
        user     = os.environ.get("PCS_SMTP_USER",     "").strip() or None
        password = os.environ.get("PCS_SMTP_PASSWORD", "").strip() or None
        use_tls  = (
            os.environ.get("PCS_SMTP_START_TLS", os.environ.get("PCS_SMTP_TLS", "1"))
        ) == "1"

        msg     = self._build_message(notification)
        to_addr = os.environ.get("PCS_SMTP_TO", msg["From"])

        log.info(
            "[smtp-driver] connecting to %s:%d tls=%s", host, port, use_tls
        )

        with smtplib.SMTP(host, port, timeout=10) as server:
            server.ehlo()
            if use_tls:
                server.starttls()
                server.ehlo()
            if user and password:
                server.login(user, password)
            server.sendmail(msg["From"], to_addr.split(","), msg.as_string())

        log.info(
            "[smtp-driver] sent alert=%s to=%s",
            notification.get("alert_type"), to_addr,
        )


# ── SMS driver (pluggable) ─────────────────────────────────────────────────────


class SmsDriverBase(NotificationDriver, ABC):
    """
    Abstract base class for SMS delivery adapters.

    No vendor credentials are baked in.  To integrate an SMS service:

    1. Subclass ``SmsDriverBase``.
    2. Implement :meth:`send` using your vendor's SDK or REST API.
    3. Pass an instance to :class:`NotificationDispatcher` (or point
       ``PCS_SMS_DRIVER`` to its fully-qualified class path and the app
       will load it automatically via :func:`load_sms_driver`).

    Example (Twilio)::

        class TwilioSmsDriver(SmsDriverBase):
            name = "sms"

            def __init__(self):
                from twilio.rest import Client
                self._client = Client(
                    os.environ["TWILIO_ACCOUNT_SID"],
                    os.environ["TWILIO_AUTH_TOKEN"],
                )
                self._from = os.environ["TWILIO_FROM"]
                self._to   = os.environ["TWILIO_TO"].split(",")

            def send(self, notification):
                payload = notification.get("payload") or {}
                body = (
                    f"[PCS ALERT] {payload.get('alert_type', '?')} "
                    f"camera={payload.get('camera_id', '?')}"
                )
                for recipient in self._to:
                    self._client.messages.create(
                        body=body, from_=self._from, to=recipient
                    )
    """

    name = "sms"

    @abstractmethod
    def send(self, notification: dict[str, Any]) -> None: ...


class NullSmsDriver(SmsDriverBase):
    """
    Default no-op SMS driver.

    Logs the would-be SMS at DEBUG level so behaviour is visible in logs
    without calling any external service or requiring vendor credentials.
    Replace with a concrete :class:`SmsDriverBase` subclass.
    """

    name = "sms"

    def is_configured(self) -> bool:
        return False   # signals dispatcher to log a warning but not raise

    def send(self, notification: dict[str, Any]) -> None:
        payload    = notification.get("payload") or {}
        alert_type = payload.get("alert_type") or notification.get("alert_type", "?")
        camera_id  = payload.get("camera_id", "N/A")
        severity   = notification.get("severity", "?")
        log.info(
            "[sms-null] Would send SMS: [%s/%s] alert_type=%s camera=%s "
            "(configure PCS_SMS_DRIVER to enable real delivery)",
            severity.upper(), alert_type, alert_type, camera_id,
        )


# ── Dynamic SMS driver loader ──────────────────────────────────────────────────


def load_sms_driver(driver_path: str) -> SmsDriverBase:
    """
    Dynamically load an SMS driver class from a dotted import path.

    The path must point to a **class** (not an instance) that is a subclass of
    :class:`SmsDriverBase`.  The class is instantiated with no arguments.

    Parameters
    ----------
    driver_path:
        Fully-qualified class path, e.g.
        ``"mypackage.sms.TwilioSmsDriver"``.

    Returns
    -------
    SmsDriverBase
        A freshly-instantiated SMS driver.

    Raises
    ------
    ImportError, AttributeError
        If the path cannot be resolved.
    TypeError
        If the class is not a subclass of :class:`SmsDriverBase`.
    """
    import importlib

    module_path, cls_name = driver_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls    = getattr(module, cls_name)

    if not (isinstance(cls, type) and issubclass(cls, SmsDriverBase)):
        raise TypeError(
            f"{driver_path!r} must be a subclass of SmsDriverBase, got {cls!r}"
        )
    return cls()
