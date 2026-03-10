"""
test_access_control.py – Unit tests for AlertStore and AccessControlEngine.

Coverage
--------
AlertStore
    - add / get / list_recent / acknowledge / count_unacked
    - filtering by alert_type, door_id, unacked_only
    - meta JSON round-trip

AccessControlEngine
    - forced_entry: door OPEN then failed verify within window
    - forced_entry: failed verify then door OPEN within window
    - no forced_entry when verify is matched
    - no forced_entry when events are outside the time window
    - tailgate: 2 persons at same door, one unmatched
    - tailgate: all matched → no alert
    - tailgate: only 1 person → no alert
    - deduplication: same door-open event does not fire twice
    - on_alert callback is invoked correctly
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.privacy.access_control import AlertStore, AccessControlEngine


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_db(tmp_path: Path) -> str:
    """Return path to a fresh temp SQLite file for each test."""
    return str(tmp_path / "test_alerts.db")


@pytest.fixture()
def alert_store(tmp_db: str) -> AlertStore:
    return AlertStore(tmp_db)


def _ts(offset_s: float = 0.0) -> str:
    """ISO-8601 UTC timestamp offset from an arbitrary fixed reference."""
    ref = datetime(2026, 2, 28, 10, 0, 0, tzinfo=timezone.utc)
    return (ref + timedelta(seconds=offset_s)).isoformat()


# ─── AlertStore tests ─────────────────────────────────────────────────────────


class TestAlertStore:
    def test_add_and_get(self, alert_store: AlertStore):
        alert = alert_store.add(
            "forced_entry",
            camera_id="gate-north",
            door_id="door-01",
            subject_key="key-abc",
            confidence=0.85,
            timestamp_utc=_ts(),
            meta={"trigger": "test"},
        )
        assert alert["id"] is not None
        assert alert["alert_type"] == "forced_entry"
        assert alert["camera_id"] == "gate-north"
        assert alert["door_id"] == "door-01"
        assert alert["subject_key"] == "key-abc"
        assert abs(alert["confidence"] - 0.85) < 1e-4
        assert alert["acknowledged"] is False
        assert alert["meta"]["trigger"] == "test"

        fetched = alert_store.get(alert["id"])
        assert fetched["id"] == alert["id"]

    def test_get_nonexistent_returns_none(self, alert_store: AlertStore):
        assert alert_store.get(99999) is None

    def test_acknowledge(self, alert_store: AlertStore):
        a = alert_store.add("tailgate", door_id="door-02", timestamp_utc=_ts())
        assert not a["acknowledged"]
        assert a["acked_by"] is None
        assert a["acked_at"] is None
        ok = alert_store.acknowledge(a["id"], acked_by="Officer Smith")
        assert ok is True
        fetched = alert_store.get(a["id"])
        assert fetched["acknowledged"] is True
        assert fetched["acked_by"] == "Officer Smith"
        assert fetched["acked_at"] is not None  # ISO-8601 UTC string

    def test_acknowledge_no_operator(self, alert_store: AlertStore):
        """Acking without an operator name leaves acked_by as None."""
        a = alert_store.add("forced_entry", timestamp_utc=_ts())
        alert_store.acknowledge(a["id"])          # no acked_by argument
        fetched = alert_store.get(a["id"])
        assert fetched["acknowledged"] is True
        assert fetched["acked_by"] is None
        assert fetched["acked_at"] is not None    # timestamp always set

    def test_new_alert_has_audit_fields(self, alert_store: AlertStore):
        """Every freshly-added alert dict contains acked_by and acked_at keys."""
        a = alert_store.add("forced_entry", timestamp_utc=_ts())
        assert "acked_by" in a
        assert "acked_at" in a
        assert a["acked_by"] is None
        assert a["acked_at"] is None

    def test_acknowledge_nonexistent_returns_false(self, alert_store: AlertStore):
        assert alert_store.acknowledge(99999) is False

    def test_count_unacked(self, alert_store: AlertStore):
        assert alert_store.count_unacked() == 0
        a1 = alert_store.add("forced_entry", timestamp_utc=_ts(0))
        a2 = alert_store.add("tailgate",     timestamp_utc=_ts(1))
        assert alert_store.count_unacked() == 2
        alert_store.acknowledge(a1["id"])
        assert alert_store.count_unacked() == 1
        alert_store.acknowledge(a2["id"])
        assert alert_store.count_unacked() == 0

    def test_list_recent_newest_first(self, alert_store: AlertStore):
        alert_store.add("forced_entry", timestamp_utc=_ts(0))
        alert_store.add("tailgate",     timestamp_utc=_ts(1))
        alert_store.add("forced_entry", timestamp_utc=_ts(2))

        rows = alert_store.list_recent(n=10)
        assert len(rows) == 3
        # Newest first (highest id first)
        assert rows[0]["id"] > rows[1]["id"] > rows[2]["id"]

    def test_list_recent_filter_by_type(self, alert_store: AlertStore):
        alert_store.add("forced_entry", door_id="d1", timestamp_utc=_ts())
        alert_store.add("tailgate",     door_id="d1", timestamp_utc=_ts())
        rows = alert_store.list_recent(alert_type="forced_entry")
        assert all(r["alert_type"] == "forced_entry" for r in rows)
        assert len(rows) == 1

    def test_list_recent_filter_by_door(self, alert_store: AlertStore):
        alert_store.add("forced_entry", door_id="d1", timestamp_utc=_ts())
        alert_store.add("forced_entry", door_id="d2", timestamp_utc=_ts())
        rows = alert_store.list_recent(door_id="d1")
        assert all(r["door_id"] == "d1" for r in rows)
        assert len(rows) == 1

    def test_list_recent_unacked_only(self, alert_store: AlertStore):
        a = alert_store.add("forced_entry", timestamp_utc=_ts())
        alert_store.add("tailgate",          timestamp_utc=_ts())
        alert_store.acknowledge(a["id"])
        rows = alert_store.list_recent(unacked_only=True)
        assert len(rows) == 1
        assert rows[0]["alert_type"] == "tailgate"

    def test_list_recent_respects_n(self, alert_store: AlertStore):
        for i in range(10):
            alert_store.add("forced_entry", timestamp_utc=_ts(i))
        rows = alert_store.list_recent(n=4)
        assert len(rows) == 4

    def test_meta_round_trip(self, alert_store: AlertStore):
        meta = {"time_diff_s": 1.23, "nested": {"a": 1}}
        a = alert_store.add("forced_entry", meta=meta, timestamp_utc=_ts())
        fetched = alert_store.get(a["id"])
        assert fetched["meta"]["time_diff_s"] == pytest.approx(1.23)
        assert fetched["meta"]["nested"]["a"] == 1

    def test_add_without_meta(self, alert_store: AlertStore):
        a = alert_store.add("forced_entry", timestamp_utc=_ts())
        assert a["meta"] == {}

    def test_default_timestamp_fills_in(self, alert_store: AlertStore):
        a = alert_store.add("forced_entry")
        assert a["timestamp_utc"] is not None
        # Should be parseable
        datetime.fromisoformat(a["timestamp_utc"].rstrip("Z"))


# ─── AccessControlEngine tests ────────────────────────────────────────────────


class TestAccessControlEngine:

    def _make_engine(self, tmp_db: str, on_alert=None):
        store = AlertStore(tmp_db)
        return store, AccessControlEngine(
            alert_store=store,
            on_alert=on_alert,
            forced_entry_window_s=2.0,
            tailgate_window_s=5.0,
            buffer_ttl_s=30.0,
        )

    # ── Forced entry ──────────────────────────────────────────────────────────

    def test_forced_entry_door_open_then_verify(self, tmp_db: str):
        """Door OPEN at t=0, failed verify at t=1 → forced_entry"""
        store, engine = self._make_engine(tmp_db)

        engine.on_verify_result(
            camera_id="gate-north",
            door_id="door-01",
            subject_key="key-abc",
            matched=False,
            similarity=0.1,
            timestamp_utc=_ts(0),
        )
        alert = engine.on_door_event("door-01", "open", _ts(1), camera_id="gate-north")

        assert alert is not None
        assert alert["alert_type"] == "forced_entry"
        assert alert["door_id"] == "door-01"
        assert store.count_unacked() == 1

    def test_forced_entry_verify_then_door_open(self, tmp_db: str):
        """Verify fails at t=0, door OPEN at t=0.5 → forced_entry"""
        store, engine = self._make_engine(tmp_db)

        engine.on_door_event("door-01", "open", _ts(0), camera_id="gate-north")

        alerts = engine.on_verify_result(
            camera_id="gate-north",
            door_id="door-01",
            subject_key="key-abc",
            matched=False,
            similarity=0.05,
            timestamp_utc=_ts(0.5),
        )
        assert len(alerts) >= 1
        assert any(a["alert_type"] == "forced_entry" for a in alerts)

    def test_forced_entry_not_fired_when_matched(self, tmp_db: str):
        """Matched verify + door open → no forced_entry."""
        store, engine = self._make_engine(tmp_db)

        engine.on_door_event("door-01", "open", _ts(0))

        alerts = engine.on_verify_result(
            camera_id="gate-north",
            door_id="door-01",
            subject_key="key-authorized",
            matched=True,
            similarity=0.92,
            timestamp_utc=_ts(0.5),
        )
        forced = [a for a in alerts if a["alert_type"] == "forced_entry"]
        assert forced == []
        assert store.count_unacked() == 0

    def test_forced_entry_outside_window_no_alert(self, tmp_db: str):
        """Door open at t=0, failed verify at t=10 → outside 2s window, no alert."""
        store, engine = self._make_engine(tmp_db)

        engine.on_door_event("door-01", "open", _ts(0), camera_id="gate-north")

        alerts = engine.on_verify_result(
            camera_id="gate-north",
            door_id="door-01",
            subject_key="key-abc",
            matched=False,
            similarity=0.1,
            timestamp_utc=_ts(10),  # 10 s later — outside ±2 s window
        )
        forced = [a for a in alerts if a["alert_type"] == "forced_entry"]
        assert forced == []

    def test_forced_entry_dedup_same_door_open(self, tmp_db: str):
        """The same door-open event should not produce two forced_entry alerts."""
        store, engine = self._make_engine(tmp_db)

        engine.on_door_event("door-01", "open", _ts(0), camera_id="gate-north")

        alerts1 = engine.on_verify_result(
            camera_id="gate-north", door_id="door-01",
            subject_key="k1", matched=False, similarity=0.1, timestamp_utc=_ts(0.5),
        )
        alerts2 = engine.on_verify_result(
            camera_id="gate-north", door_id="door-01",
            subject_key="k2", matched=False, similarity=0.1, timestamp_utc=_ts(1.0),
        )
        # Only the first should produce the forced_entry alert for this door-open
        forced1 = [a for a in alerts1 if a["alert_type"] == "forced_entry"]
        forced2 = [a for a in alerts2 if a["alert_type"] == "forced_entry"]
        assert len(forced1) + len(forced2) == 1

    def test_forced_entry_door_close_no_alert(self, tmp_db: str):
        """Door CLOSE event + failed verify → no forced_entry (only OPEN triggers it)."""
        store, engine = self._make_engine(tmp_db)

        engine.on_verify_result(
            camera_id=None, door_id="door-01", subject_key=None,
            matched=False, similarity=0.1, timestamp_utc=_ts(0),
        )
        alert = engine.on_door_event("door-01", "close", _ts(0.5))
        assert alert is None
        assert store.count_unacked() == 0

    def test_forced_entry_no_door_id_skipped(self, tmp_db: str):
        """Verify result with no door_id is not correlated."""
        store, engine = self._make_engine(tmp_db)

        engine.on_door_event("door-01", "open", _ts(0))
        alerts = engine.on_verify_result(
            camera_id="gate-north", door_id=None,  # no door_id given
            subject_key="k1", matched=False, similarity=0.1, timestamp_utc=_ts(0.5),
        )
        assert alerts == []

    # ── Tailgate ──────────────────────────────────────────────────────────────

    def test_tailgate_two_persons_one_unmatched(self, tmp_db: str):
        """Authorized person followed by unmatched one within 5 s → tailgate."""
        store, engine = self._make_engine(tmp_db)

        engine.on_verify_result(
            camera_id="gate-north", door_id="door-01",
            subject_key="auth-key", matched=True, similarity=0.9,
            timestamp_utc=_ts(0),
        )
        alerts = engine.on_verify_result(
            camera_id="gate-north", door_id="door-01",
            subject_key="unauth-key", matched=False, similarity=0.05,
            timestamp_utc=_ts(3),
        )
        tailgate = [a for a in alerts if a["alert_type"] == "tailgate"]
        assert len(tailgate) == 1
        assert tailgate[0]["door_id"] == "door-01"
        assert store.count_unacked() == 1

    def test_tailgate_two_persons_both_matched_no_alert(self, tmp_db: str):
        """Two authorized people close together → no tailgate."""
        store, engine = self._make_engine(tmp_db)

        engine.on_verify_result(
            camera_id="gate-north", door_id="door-01",
            subject_key="auth-1", matched=True, similarity=0.9, timestamp_utc=_ts(0),
        )
        alerts = engine.on_verify_result(
            camera_id="gate-north", door_id="door-01",
            subject_key="auth-2", matched=True, similarity=0.88, timestamp_utc=_ts(2),
        )
        tailgate = [a for a in alerts if a["alert_type"] == "tailgate"]
        assert tailgate == []

    def test_tailgate_single_person_no_alert(self, tmp_db: str):
        """Single person (even unmatched) does not trigger tailgate."""
        store, engine = self._make_engine(tmp_db)

        alerts = engine.on_verify_result(
            camera_id="gate-north", door_id="door-01",
            subject_key="k1", matched=False, similarity=0.05, timestamp_utc=_ts(0),
        )
        tailgate = [a for a in alerts if a["alert_type"] == "tailgate"]
        assert tailgate == []

    def test_tailgate_outside_window_no_alert(self, tmp_db: str):
        """The second person arrives 10 s later → outside 5 s window, no tailgate."""
        store, engine = self._make_engine(tmp_db)

        engine.on_verify_result(
            camera_id="gate-north", door_id="door-01",
            subject_key="auth-1", matched=True, similarity=0.9, timestamp_utc=_ts(0),
        )
        alerts = engine.on_verify_result(
            camera_id="gate-north", door_id="door-01",
            subject_key="unauth-1", matched=False, similarity=0.1, timestamp_utc=_ts(10),
        )
        tailgate = [a for a in alerts if a["alert_type"] == "tailgate"]
        assert tailgate == []

    def test_tailgate_no_door_id_skipped(self, tmp_db: str):
        """Verify results without door_id are not used for tailgate correlation."""
        store, engine = self._make_engine(tmp_db)

        engine.on_verify_result(
            camera_id="gate-north", door_id=None,
            subject_key="auth-1", matched=True, similarity=0.9, timestamp_utc=_ts(0),
        )
        alerts = engine.on_verify_result(
            camera_id="gate-north", door_id=None,
            subject_key="unauth-1", matched=False, similarity=0.05, timestamp_utc=_ts(3),
        )
        tailgate = [a for a in alerts if a["alert_type"] == "tailgate"]
        assert tailgate == []

    # ── on_alert callback ─────────────────────────────────────────────────────

    def test_on_alert_callback_invoked(self, tmp_db: str):
        """The on_alert callback must be called with the alert dict."""
        received: list[dict] = []
        store, engine = self._make_engine(tmp_db, on_alert=received.append)

        engine.on_door_event("door-01", "open", _ts(0))
        engine.on_verify_result(
            camera_id=None, door_id="door-01",
            subject_key=None, matched=False, similarity=0.1, timestamp_utc=_ts(0.5),
        )
        assert len(received) >= 1
        assert received[0]["alert_type"] == "forced_entry"

    def test_on_alert_callback_exception_does_not_propagate(self, tmp_db: str):
        """A raising callback must not crash the engine."""
        def _bad_callback(alert):
            raise RuntimeError("boom")

        store, engine = self._make_engine(tmp_db, on_alert=_bad_callback)

        engine.on_door_event("door-01", "open", _ts(0))
        # Should not raise:
        engine.on_verify_result(
            camera_id=None, door_id="door-01",
            subject_key=None, matched=False, similarity=0.05, timestamp_utc=_ts(0.5),
        )
        assert store.count_unacked() == 1  # alert was still stored

    # ── Multiple doors ────────────────────────────────────────────────────────

    def test_different_doors_not_correlated(self, tmp_db: str):
        """Door-01 open should not correlate with a failed verify at door-02."""
        store, engine = self._make_engine(tmp_db)

        engine.on_door_event("door-01", "open", _ts(0))
        alerts = engine.on_verify_result(
            camera_id=None, door_id="door-02",  # different door
            subject_key="k1", matched=False, similarity=0.1, timestamp_utc=_ts(0.5),
        )
        forced = [a for a in alerts if a["alert_type"] == "forced_entry"]
        assert forced == []
