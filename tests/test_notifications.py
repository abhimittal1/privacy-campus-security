"""
tests/test_notifications.py – Unit & integration tests for the
notifications subsystem (store, policy, drivers, dispatcher).

Coverage
--------
NotificationStore        – CRUD, pending retry, list/filter
EscalationPolicy         – severity map, channel resolution, custom overrides
WebSocketDriver          – mock socketio emit, correct event routing
SmtpDriver               – not-configured guard, message formatting
NullSmsDriver            – no-op, is_configured=False
NotificationDispatcher   – dispatch flow, throttle, retry worker, async delivery
Flask integration        – GET /api/notifications; high-severity trigger
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.notifications.store      import NotificationStore
from src.notifications.policy     import EscalationPolicy, SEVERITY_MAP
from src.notifications.drivers    import (
    WebSocketDriver,
    SmtpDriver,
    SmtpNotConfiguredError,
    NullSmsDriver,
    SmsDriverBase,
    load_sms_driver,
)
from src.notifications.dispatcher import NotificationDispatcher


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _tmp_store() -> NotificationStore:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return NotificationStore(path)


def _alert(alert_type: str = "safety_alert", alert_id: int = 1) -> dict:
    return {
        "id":           alert_id,
        "alert_type":   alert_type,
        "camera_id":    "gate-north",
        "door_id":      "door-01",
        "confidence":   0.95,
        "timestamp_utc": "2026-03-01T10:00:00+00:00",
        "meta":         {"severity": "high"},
        "acknowledged": False,
    }


def _make_dispatcher(
    store: NotificationStore,
    *,
    policy=None,
    extra_drivers=None,
    throttle_s: float = 0.0,  # disable throttle in most tests
    max_retries: int  = 2,
) -> tuple[NotificationDispatcher, MagicMock]:
    """Return (dispatcher, mock_socketio) for testing."""
    mock_sio = MagicMock()
    drivers = [WebSocketDriver(mock_sio), SmtpDriver(), NullSmsDriver()]
    if extra_drivers:
        drivers.extend(extra_drivers)
    if policy is None:
        policy = EscalationPolicy()
    d = NotificationDispatcher(
        store       = store,
        policy      = policy,
        drivers     = drivers,
        throttle_s  = throttle_s,
        max_retries = max_retries,
    )
    return d, mock_sio


# ═════════════════════════════════════════════════════════════════════════════
#  NotificationStore
# ═════════════════════════════════════════════════════════════════════════════


class TestNotificationStore:

    def test_add_returns_dict_with_id(self):
        store = _tmp_store()
        n = store.add(alert_id=1, alert_type="safety_alert",
                      severity="high", channel="websocket")
        assert isinstance(n, dict)
        assert n["id"] >= 1
        assert n["alert_type"] == "safety_alert"
        assert n["channel"] == "websocket"
        assert n["status"] == "pending"
        assert n["attempts"] == 0

    def test_add_custom_status(self):
        store = _tmp_store()
        n = store.add(alert_id=2, alert_type="loitering",
                      severity="medium", channel="smtp", status="throttled")
        assert n["status"] == "throttled"

    def test_add_with_payload_roundtrips(self):
        store = _tmp_store()
        payload = {"alert_type": "forced_entry", "camera_id": "cam-1"}
        n = store.add(alert_id=3, alert_type="forced_entry",
                      severity="high", channel="websocket", payload=payload)
        fetched = store.get(n["id"])
        assert fetched is not None
        assert fetched["payload"] == payload

    def test_get_unknown_id_returns_none(self):
        store = _tmp_store()
        assert store.get(99999) is None

    def test_update_status_sent(self):
        store = _tmp_store()
        n = store.add(alert_id=1, alert_type="tailgate",
                      severity="high", channel="websocket")
        store.update_status(n["id"], "sent")
        fetched = store.get(n["id"])
        assert fetched["status"] == "sent"
        assert fetched["last_attempt"] is not None

    def test_update_status_failed_with_error_and_retry(self):
        store = _tmp_store()
        n = store.add(alert_id=1, alert_type="tailgate",
                      severity="high", channel="smtp")
        store.update_status(n["id"], "failed",
                            error_msg="conn refused",
                            next_retry="2026-03-01T10:00:30+00:00")
        fetched = store.get(n["id"])
        assert fetched["status"] == "failed"
        assert fetched["error_msg"] == "conn refused"
        assert fetched["next_retry"] == "2026-03-01T10:00:30+00:00"

    def test_increment_attempts(self):
        store = _tmp_store()
        n = store.add(alert_id=1, alert_type="safety_alert",
                      severity="high", channel="sms")
        store.increment_attempts(n["id"])
        store.increment_attempts(n["id"])
        fetched = store.get(n["id"])
        assert fetched["attempts"] == 2

    def test_list_recent_newest_first(self):
        store = _tmp_store()
        for i in range(5):
            store.add(alert_id=i, alert_type="loitering",
                      severity="medium", channel="websocket")
        records = store.list_recent(n=10)
        ids = [r["id"] for r in records]
        assert ids == sorted(ids, reverse=True)

    def test_list_recent_filters_by_channel(self):
        store = _tmp_store()
        store.add(alert_id=1, alert_type="safety_alert",
                  severity="high", channel="websocket")
        store.add(alert_id=1, alert_type="safety_alert",
                  severity="high", channel="smtp")
        ws_records = store.list_recent(channel="websocket")
        assert all(r["channel"] == "websocket" for r in ws_records)
        assert len(ws_records) == 1

    def test_list_recent_filters_by_status(self):
        store = _tmp_store()
        n1 = store.add(alert_id=1, alert_type="tailgate",
                       severity="high", channel="websocket")
        store.update_status(n1["id"], "sent")
        n2 = store.add(alert_id=2, alert_type="tailgate",
                       severity="high", channel="smtp")
        sent = store.list_recent(status="sent")
        assert all(r["status"] == "sent" for r in sent)
        assert len(sent) == 1

    def test_list_recent_filters_by_alert_id(self):
        store = _tmp_store()
        store.add(alert_id=10, alert_type="safety_alert",
                  severity="high", channel="websocket")
        store.add(alert_id=99, alert_type="safety_alert",
                  severity="high", channel="smtp")
        records = store.list_recent(alert_id=10)
        assert all(r["alert_id"] == 10 for r in records)
        assert len(records) == 1

    def test_list_recent_filters_by_alert_type(self):
        store = _tmp_store()
        store.add(alert_id=1, alert_type="loitering",
                  severity="medium", channel="websocket")
        store.add(alert_id=2, alert_type="safety_alert",
                  severity="high", channel="websocket")
        records = store.list_recent(alert_type="loitering")
        assert all(r["alert_type"] == "loitering" for r in records)
        assert len(records) == 1

    def test_get_pending_retry_excludes_sent(self):
        store = _tmp_store()
        n = store.add(alert_id=1, alert_type="safety_alert",
                      severity="high", channel="smtp")
        store.update_status(n["id"], "sent")
        pending = store.get_pending_retry(max_retries=3)
        assert all(p["id"] != n["id"] for p in pending)

    def test_get_pending_retry_picks_up_failed_with_past_retry(self):
        store = _tmp_store()
        n = store.add(alert_id=1, alert_type="tailgate",
                      severity="high", channel="smtp")
        # Simulate one failed attempt with a past retry time
        store.increment_attempts(n["id"])
        store.update_status(n["id"], "failed",
                            error_msg="timeout",
                            next_retry="2020-01-01T00:00:00+00:00")
        pending = store.get_pending_retry(max_retries=3)
        assert any(p["id"] == n["id"] for p in pending)

    def test_get_pending_retry_excludes_future_retry(self):
        store = _tmp_store()
        n = store.add(alert_id=1, alert_type="tailgate",
                      severity="high", channel="smtp")
        store.increment_attempts(n["id"])
        store.update_status(n["id"], "failed",
                            next_retry="2099-01-01T00:00:00+00:00")
        pending = store.get_pending_retry(max_retries=3)
        assert all(p["id"] != n["id"] for p in pending)

    def test_get_pending_retry_excludes_max_retries_reached(self):
        store = _tmp_store()
        n = store.add(alert_id=1, alert_type="tailgate",
                      severity="high", channel="smtp")
        # Simulate 3 failed attempts (at max_retries=3 they should not be retried)
        for _ in range(3):
            store.increment_attempts(n["id"])
        store.update_status(n["id"], "failed",
                            next_retry="2020-01-01T00:00:00+00:00")
        pending = store.get_pending_retry(max_retries=3)
        assert all(p["id"] != n["id"] for p in pending)

    def test_count_by_status(self):
        store = _tmp_store()
        n1 = store.add(alert_id=1, alert_type="loitering",
                       severity="medium", channel="websocket")
        n2 = store.add(alert_id=2, alert_type="loitering",
                       severity="medium", channel="smtp")
        store.update_status(n1["id"], "sent")
        assert store.count_by_status("sent")    == 1
        assert store.count_by_status("pending") == 1
        assert store.count_by_status("failed")  == 0

    def test_thread_safety(self):
        """Multiple threads can write concurrently without corruption."""
        store = _tmp_store()
        errors: list[Exception] = []

        def _writer():
            try:
                for _ in range(20):
                    store.add(alert_id=1, alert_type="tailgate",
                              severity="high", channel="websocket")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_writer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(store.list_recent(n=200)) == 100  # 5 threads × 20 records


# ═════════════════════════════════════════════════════════════════════════════
#  EscalationPolicy
# ═════════════════════════════════════════════════════════════════════════════


class TestEscalationPolicy:

    def test_default_severity_map_high(self):
        p = EscalationPolicy()
        for atype in ("safety_alert", "forced_entry", "tailgate"):
            assert p.get_severity(atype) == "high", f"{atype} should be high"

    def test_default_severity_map_medium(self):
        p = EscalationPolicy()
        for atype in ("loitering", "left_object"):
            assert p.get_severity(atype) == "medium", f"{atype} should be medium"

    def test_unknown_type_defaults_to_medium(self):
        p = EscalationPolicy()
        assert p.get_severity("banana") == "medium"

    def test_channels_low(self):
        p = EscalationPolicy()
        assert p.get_channels("low") == ["websocket"]

    def test_channels_medium(self):
        p = EscalationPolicy()
        ch = p.get_channels("medium")
        assert "websocket" in ch
        assert "smtp" in ch
        assert "sms" not in ch

    def test_channels_high_without_sms_opt_in(self):
        p = EscalationPolicy(sms_opt_in=False)
        ch = p.get_channels("high")
        assert "websocket" in ch
        assert "smtp"      in ch
        assert "sms" not in ch

    def test_channels_high_with_sms_opt_in(self):
        p = EscalationPolicy(sms_opt_in=True)
        ch = p.get_channels("high")
        assert "websocket" in ch
        assert "smtp"      in ch
        assert "sms"       in ch

    def test_unknown_severity_falls_back_to_websocket_only(self):
        p = EscalationPolicy()
        assert p.get_channels("critical") == ["websocket"]

    def test_custom_policy_override(self):
        custom = {"high": ["websocket"], "medium": [], "low": []}
        p = EscalationPolicy(policy=custom)
        assert p.get_channels("high")   == ["websocket"]
        assert p.get_channels("medium") == []

    def test_custom_severity_map_override(self):
        custom_map = {**SEVERITY_MAP, "loitering": "high"}
        p = EscalationPolicy(severity_map=custom_map)
        assert p.get_severity("loitering") == "high"

    def test_channels_for_alert_convenience(self):
        p = EscalationPolicy(sms_opt_in=False)
        ch = p.channels_for_alert("safety_alert")
        assert "websocket" in ch
        assert "smtp"      in ch


# ═════════════════════════════════════════════════════════════════════════════
#  Drivers
# ═════════════════════════════════════════════════════════════════════════════


class TestWebSocketDriver:

    def _notif(self, alert_type: str, payload: dict | None = None) -> dict:
        p = payload or {"alert_type": alert_type, "camera_id": "cam-1"}
        return {
            "id":         1,
            "alert_type": alert_type,
            "channel":    "websocket",
            "payload":    p,
        }

    def test_sends_alert_event_for_safety_alert(self):
        sio = MagicMock()
        drv = WebSocketDriver(sio)
        drv.send(self._notif("safety_alert"))
        sio.emit.assert_called_once()
        event, payload = sio.emit.call_args[0]
        assert event == "alert"

    def test_sends_alert_event_for_forced_entry(self):
        sio = MagicMock()
        drv = WebSocketDriver(sio)
        drv.send(self._notif("forced_entry"))
        event_name = sio.emit.call_args[0][0]
        assert event_name == "alert"

    def test_sends_alert_event_for_tailgate(self):
        sio = MagicMock()
        drv = WebSocketDriver(sio)
        drv.send(self._notif("tailgate"))
        event_name = sio.emit.call_args[0][0]
        assert event_name == "alert"

    def test_sends_loitering_event(self):
        sio = MagicMock()
        drv = WebSocketDriver(sio)
        drv.send(self._notif("loitering"))
        event_name = sio.emit.call_args[0][0]
        assert event_name == "loitering"

    def test_sends_left_object_event(self):
        sio = MagicMock()
        drv = WebSocketDriver(sio)
        drv.send(self._notif("left_object"))
        event_name = sio.emit.call_args[0][0]
        assert event_name == "left_object"

    def test_unknown_type_defaults_to_alert_event(self):
        sio = MagicMock()
        drv = WebSocketDriver(sio)
        drv.send(self._notif("custom_event"))
        event_name = sio.emit.call_args[0][0]
        assert event_name == "alert"

    def test_payload_passed_to_emit(self):
        sio = MagicMock()
        drv = WebSocketDriver(sio)
        payload = {"alert_type": "tailgate", "camera_id": "lobby"}
        drv.send({"id": 1, "alert_type": "tailgate",
                  "channel": "websocket", "payload": payload})
        _, emitted_payload = sio.emit.call_args[0]
        assert emitted_payload == payload

    def test_is_configured_always_true(self):
        sio = MagicMock()
        assert WebSocketDriver(sio).is_configured() is True


class TestSmtpDriver:

    def test_raises_not_configured_when_no_host(self):
        drv = SmtpDriver()
        notif = {
            "id": 1, "alert_type": "safety_alert", "severity": "high",
            "channel": "smtp", "payload": {"alert_type": "safety_alert"},
        }
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PCS_SMTP_HOST", None)
            with pytest.raises(SmtpNotConfiguredError):
                drv.send(notif)

    def test_is_configured_false_without_host(self):
        drv = SmtpDriver()
        with patch.dict(os.environ, {"PCS_SMTP_HOST": ""}):
            assert drv.is_configured() is False

    def test_is_configured_true_with_host(self):
        drv = SmtpDriver()
        with patch.dict(os.environ, {"PCS_SMTP_HOST": "smtp.example.com"}):
            assert drv.is_configured() is True

    def test_send_calls_smtplib_when_configured(self):
        drv = SmtpDriver()
        notif = {
            "id": 1, "alert_type": "safety_alert", "severity": "high",
            "channel": "smtp", "created_at": "2026-03-01T10:00:00+00:00",
            "payload": {
                "alert_type":   "safety_alert",
                "camera_id":    "gate-north",
                "confidence":   0.99,
                "timestamp_utc": "2026-03-01T10:00:00+00:00",
                "meta":         {"severity": "high"},
            },
        }
        env = {
            "PCS_SMTP_HOST":    "smtp.example.com",
            "PCS_SMTP_PORT":    "587",
            "PCS_SMTP_FROM":    "pcs@example.com",
            "PCS_SMTP_TO":      "admin@example.com",
            "PCS_SMTP_TLS":     "0",  # disable TLS so we don't need real certs
        }
        with patch.dict(os.environ, env):
            with patch("smtplib.SMTP") as mock_smtp_cls:
                mock_server = MagicMock()
                mock_smtp_cls.return_value.__enter__ = lambda s: mock_server
                mock_smtp_cls.return_value.__exit__  = MagicMock(return_value=False)
                drv.send(notif)
                mock_smtp_cls.assert_called_once_with("smtp.example.com", 587, timeout=10)

    def test_build_message_includes_alert_type_in_subject(self):
        drv = SmtpDriver()
        notif = {
            "id": 1, "alert_type": "forced_entry", "severity": "high",
            "channel": "smtp", "created_at": "2026-03-01T10:00:00+00:00",
            "payload": {
                "alert_type": "forced_entry", "camera_id": "side-exit",
                "confidence": 0.92, "timestamp_utc": "2026-03-01T10:00:00+00:00",
                "meta": {},
            },
        }
        with patch.dict(os.environ, {"PCS_SMTP_FROM": "pcs@x.com"}):
            msg = drv._build_message(notif)
        assert "forced_entry" in msg["Subject"].lower() or "forced" in msg["Subject"].lower()
        assert "side-exit" in msg["Subject"]


class TestNullSmsDriver:

    def test_send_does_not_raise(self):
        drv = NullSmsDriver()
        drv.send({"id": 1, "alert_type": "safety_alert",
                  "severity": "high", "payload": {}})

    def test_is_configured_false(self):
        assert NullSmsDriver().is_configured() is False

    def test_name_is_sms(self):
        assert NullSmsDriver().name == "sms"


class TestLoadSmsDriver:

    def test_load_null_driver_from_path(self):
        drv = load_sms_driver(
            "src.notifications.drivers.NullSmsDriver"
        )
        assert isinstance(drv, NullSmsDriver)

    def test_load_nonexistent_module_raises(self):
        with pytest.raises((ImportError, ModuleNotFoundError)):
            load_sms_driver("nonexistent.module.FooDriver")

    def test_load_non_subclass_raises_type_error(self):
        """Passing a class that is not a SmsDriverBase subclass should fail."""
        with pytest.raises(TypeError):
            load_sms_driver("src.notifications.drivers.SmtpDriver")


# ═════════════════════════════════════════════════════════════════════════════
#  NotificationDispatcher
# ═════════════════════════════════════════════════════════════════════════════


class TestNotificationDispatcher:

    def test_dispatch_creates_notification_record_per_channel(self):
        store = _tmp_store()
        d, mock_sio = _make_dispatcher(store)
        alert = _alert("safety_alert")
        records = d.dispatch(alert)
        # high severity → websocket + smtp (no sms opt-in in default policy)
        channels = {r["channel"] for r in records}
        assert "websocket" in channels
        assert "smtp"      in channels

    def test_dispatch_websocket_synchronous_emit(self):
        store = _tmp_store()
        d, mock_sio = _make_dispatcher(store)
        alert = _alert("forced_entry")
        d.dispatch(alert)
        # WebSocket is dispatched synchronously → emit should have been called
        mock_sio.emit.assert_called()

    def test_dispatch_websocket_sent_status(self):
        store = _tmp_store()
        d, mock_sio = _make_dispatcher(store)
        alert = _alert("tailgate")
        records = d.dispatch(alert)
        ws_rec = next(r for r in records if r["channel"] == "websocket")
        # Refresh from DB (dispatch returns the pre-async record for non-ws)
        ws_fetched = store.get(ws_rec["id"])
        assert ws_fetched["status"] == "sent"

    def test_dispatch_medium_severity_no_sms(self):
        store = _tmp_store()
        d, mock_sio = _make_dispatcher(store)
        alert = _alert("loitering")
        records = d.dispatch(alert)
        channels = {r["channel"] for r in records}
        assert "sms" not in channels
        assert "websocket" in channels

    def test_dispatch_low_severity_websocket_only(self):
        store = _tmp_store()
        d, mock_sio = _make_dispatcher(
            store, policy=EscalationPolicy(
                policy={"low": ["websocket"], "medium": ["websocket"], "high": ["websocket"]}
            )
        )
        alert = _alert("loitering")  # medium by default
        records = d.dispatch(alert)
        assert all(r["channel"] == "websocket" for r in records)

    def test_dispatch_explicit_severity_override(self):
        store = _tmp_store()
        d, mock_sio = _make_dispatcher(store,
            policy=EscalationPolicy(sms_opt_in=True))
        alert = _alert("loitering")
        records = d.dispatch(alert, severity="high")
        channels = {r["channel"] for r in records}
        assert "sms" in channels

    def test_throttle_blocks_duplicate(self):
        """Two dispatches within throttle window produce 2nd as throttled."""
        store = _tmp_store()
        d, mock_sio = _make_dispatcher(store, throttle_s=9999.0)

        alert = _alert("safety_alert")
        # First dispatch
        recs1 = d.dispatch(alert)
        ws1 = next(r for r in recs1 if r["channel"] == "websocket")
        # Give the async SMTP thread a moment
        time.sleep(0.05)
        ws1_status = store.get(ws1["id"])["status"]
        assert ws1_status == "sent"  # first one went through

        # Second dispatch — should be throttled for websocket
        recs2 = d.dispatch(alert)
        ws2 = next(r for r in recs2 if r["channel"] == "websocket")
        assert ws2["status"] == "throttled"

    def test_throttle_zero_disables_throttling(self):
        """With throttle_s=0, same alert dispatched twice creates two sent records."""
        store = _tmp_store()
        d, mock_sio = _make_dispatcher(store, throttle_s=0.0)
        alert = _alert("safety_alert")
        d.dispatch(alert)
        time.sleep(0.05)
        d.dispatch(alert)
        ws_records = store.list_recent(channel="websocket", alert_type="safety_alert")
        sent = [r for r in ws_records if r["status"] == "sent"]
        assert len(sent) == 2

    def test_smtp_not_configured_marks_failed(self):
        """SMTP delivery fails gracefully when PCS_SMTP_HOST is absent."""
        store = _tmp_store()
        d, mock_sio = _make_dispatcher(store)
        alert = _alert("safety_alert")
        env = dict(os.environ)
        env.pop("PCS_SMTP_HOST", None)
        with patch.dict(os.environ, env, clear=True):
            d.dispatch(alert)
        # Give the async SMTP thread time to finish
        time.sleep(0.2)
        smtp_recs = store.list_recent(channel="smtp")
        # Should be exactly one SMTP record, marked failed
        assert any(r["status"] == "failed" for r in smtp_recs)

    def test_retry_worker_retries_failed_notification(self):
        """A failed record with a past next_retry is re-attempted and eventually sent."""
        store = _tmp_store()

        # Driver always succeeds — we just want to see that the retry worker
        # picks up the pre-seeded failed record and marks it sent.
        failing_driver = MagicMock()
        failing_driver.name = "smtp"
        failing_driver.send.return_value = None  # always succeeds

        policy = EscalationPolicy(
            policy={"high": ["smtp"], "medium": ["smtp"], "low": []}
        )
        d = NotificationDispatcher(
            store            = store,
            policy           = policy,
            drivers          = [failing_driver],
            throttle_s       = 0.0,
            max_retries      = 3,
            retry_interval_s = 0.05,   # fast sweep for test
        )

        # Pre-seed: a record that already failed once with a past next_retry
        n = store.add(
            alert_id   = 1,
            alert_type = "safety_alert",
            severity   = "high",
            channel    = "smtp",
            payload    = {"alert_type": "safety_alert", "camera_id": "gate-north"},
        )
        store.increment_attempts(n["id"])
        store.update_status(
            n["id"], "failed",
            error_msg  = "conn reset",
            next_retry = "2020-01-01T00:00:00+00:00",   # already past
        )

        d.start()
        time.sleep(0.4)  # allow retry worker to fire

        smtp_recs = store.list_recent(channel="smtp", alert_type="safety_alert")
        assert any(r["status"] == "sent" for r in smtp_recs), \
            f"Expected a sent record among: {smtp_recs}"

    def test_retry_gives_up_after_max_retries(self):
        """Permanently failing driver is retried max_retries-1 times then abandoned."""
        store = _tmp_store()
        failing_driver = MagicMock()
        failing_driver.name = "smtp"
        failing_driver.send.side_effect = OSError("always fails")

        policy = EscalationPolicy(
            policy={"high": ["smtp"], "medium": ["smtp"], "low": []}
        )
        d = NotificationDispatcher(
            store            = store,
            policy           = policy,
            drivers          = [failing_driver],
            throttle_s       = 0.0,
            max_retries      = 2,      # 2 total attempts allowed
            retry_interval_s = 0.05,
        )

        # Pre-seed: 1 attempt already made; one retry attempt is still allowed
        n = store.add(
            alert_id   = 1,
            alert_type = "safety_alert",
            severity   = "high",
            channel    = "smtp",
            payload    = {"alert_type": "safety_alert"},
        )
        store.increment_attempts(n["id"])
        store.update_status(
            n["id"], "failed",
            error_msg  = "init fail",
            next_retry = "2020-01-01T00:00:00+00:00",
        )

        d.start()
        time.sleep(0.4)  # allow the retry worker to exhaust attempts

        smtp_recs = store.list_recent(channel="smtp", alert_type="safety_alert")
        # All should be failed (none succeeded)
        assert all(r["status"] == "failed" for r in smtp_recs)
        # 2nd attempt was made (attempts now == 2 == max_retries)
        assert all(r["attempts"] == 2 for r in smtp_recs)

    def test_dispatch_no_driver_for_channel_is_skipped(self):
        """If a channel has no registered driver, it's silently skipped."""
        store = _tmp_store()
        mock_sio = MagicMock()
        policy = EscalationPolicy(policy={"high": ["websocket", "pigeon"]},
                                  severity_map={"safety_alert": "high"})
        d = NotificationDispatcher(
            store      = store,
            policy     = policy,
            drivers    = [WebSocketDriver(mock_sio)],
            throttle_s = 0.0,
        )
        records = d.dispatch(_alert("safety_alert"))
        channels = {r["channel"] for r in records}
        assert "pigeon" not in channels
        assert "websocket" in channels

    def test_sms_opt_in_via_env(self):
        """PCS_SMS_OPT_IN=1 includes sms channel for high-severity alerts."""
        with patch.dict(os.environ, {"PCS_SMS_OPT_IN": "1"}):
            p = EscalationPolicy()
        assert "sms" in p.get_channels("high")


# ═════════════════════════════════════════════════════════════════════════════
#  Flask integration tests
# ═════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def app_client():
    """Return a Flask test client with the full application factory."""
    import tempfile, os
    _fd, _path = tempfile.mkstemp(suffix=".db")
    os.close(_fd)
    _store_fd, _store_path = tempfile.mkstemp(suffix=".json")
    os.close(_store_fd)
    os.environ.setdefault("PCS_ATTENDANCE_DB", _path)
    os.environ.setdefault("PCS_STORE",         _store_path)

    from src.web.flask_app import create_app
    _app = create_app()
    _app.config["TESTING"] = True
    with _app.test_client() as client:
        yield client


class TestFlaskNotifications:
    """Integration tests for GET /api/notifications."""

    def test_api_notifications_returns_200(self, app_client):
        resp = app_client.get("/api/notifications")
        assert resp.status_code == 200, resp.get_data(as_text=True)
        data = resp.get_json()
        assert "notifications"    in data
        assert "total_sent"       in data
        assert "total_failed"     in data
        assert "total_throttled"  in data

    def test_loitering_trigger_creates_notification(self, app_client):
        """POST /api/loitering/trigger → GET /api/notifications shows loitering record."""
        trigger_resp = app_client.post(
            "/api/loitering/trigger",
            json={"camera_id": "gate-north", "subject_key": "subj-001", "dwell_s": 65},
        )
        assert trigger_resp.status_code == 200, trigger_resp.get_data(as_text=True)
        # Allow async background threads to complete
        time.sleep(0.3)
        notif_resp = app_client.get(
            "/api/notifications?alert_type=loitering&n=10"
        )
        assert notif_resp.status_code == 200
        data = notif_resp.get_json()
        assert any(
            r["alert_type"] == "loitering"
            for r in data["notifications"]
        ), f"Expected loitering notification; got: {data['notifications']}"

    def test_left_object_trigger_creates_notification(self, app_client):
        """POST /api/left-objects/trigger → GET /api/notifications shows left_object record."""
        trigger_resp = app_client.post(
            "/api/left-objects/trigger",
            json={"camera_id": "parking-east", "dwell_s": 45},
        )
        assert trigger_resp.status_code == 200
        time.sleep(0.3)
        notif_resp = app_client.get(
            "/api/notifications?alert_type=left_object&n=10"
        )
        assert notif_resp.status_code == 200
        data = notif_resp.get_json()
        assert any(
            r["alert_type"] == "left_object"
            for r in data["notifications"]
        )

    def test_high_severity_trigger_creates_websocket_record(self, app_client):
        """Injecting a loitering event creates a websocket notification record."""
        app_client.post(
            "/api/loitering/trigger",
            json={"camera_id": "gate-north", "subject_key": "high-subj", "dwell_s": 70},
        )
        time.sleep(0.3)
        notif_resp = app_client.get("/api/notifications?channel=websocket&n=50")
        data = notif_resp.get_json()
        assert any(r["channel"] == "websocket" for r in data["notifications"])

    def test_api_notifications_filter_by_channel(self, app_client):
        """Filtering by channel returns only matching records."""
        resp = app_client.get("/api/notifications?channel=smtp&n=10")
        assert resp.status_code == 200
        data = resp.get_json()
        assert all(r["channel"] == "smtp" for r in data["notifications"])

    def test_api_notifications_filter_by_status(self, app_client):
        """Filtering by status returns only matching records."""
        resp = app_client.get("/api/notifications?status=sent&n=10")
        assert resp.status_code == 200
        data = resp.get_json()
        assert all(r["status"] == "sent" for r in data["notifications"])

    def test_api_notifications_payload_omitted_in_list(self, app_client):
        """List response omits heavy payload field; only slim fields returned."""
        resp = app_client.get("/api/notifications?n=5")
        data = resp.get_json()
        for r in data["notifications"]:
            assert "payload" not in r, "payload should be stripped from list view"

    def test_api_notifications_respects_n_cap(self, app_client):
        resp = app_client.get("/api/notifications?n=3")
        data = resp.get_json()
        assert len(data["notifications"]) <= 3
