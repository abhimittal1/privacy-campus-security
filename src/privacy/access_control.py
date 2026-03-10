"""
access_control.py – Real-time access-control engine: door event correlation,
forced-entry detection, and tailgating detection.

Overview
--------
Two objects are provided:

``AlertStore`` (SQLite-backed)
    Persists security alerts to a table that coexists with the embedding,
    attendance, and camera tables in the same SQLite file.  Exposes query,
    acknowledge, and list methods.

``AccessControlEngine``
    Stateful, thread-safe engine that receives two kinds of events:

    * **Door events** — ``on_door_event(door_id, event_type, timestamp_utc)``
      published by the MQTT agent subscribed to ``campus/<door_id>/event``.

    * **Verify results** — ``on_verify_result(...)`` called after each
      face-recognition attempt (REST or Socket.IO).

    The engine correlates these streams using short in-memory ring buffers
    and fires two alert types:

    **forced_entry**
        A door-open event occurred and, within ±2 seconds, a verify
        operation for the same door produced ``matched == False``.  Indicates
        a person tried to use an unrecognised face to open a door, or a door
        was physically forced while a non-matching face was nearby.

    **tailgate**
        Two or more distinct subjects passed through the same door within a
        5-second window, but at least one of them was unmatched.  Indicates
        an authorised person was followed by an unauthorised one.

Schema (``alerts`` table)
--------------------------
::

    id              INTEGER  PRIMARY KEY AUTOINCREMENT
    alert_type      TEXT     NOT NULL  -- 'forced_entry' | 'tailgate'
    camera_id       TEXT               -- nullable
    door_id         TEXT               -- nullable
    subject_key     TEXT               -- nullable; who triggered (or comma-joined list)
    confidence      REAL     DEFAULT 1.0
    timestamp_utc   TEXT     NOT NULL  -- ISO-8601 UTC
    meta_json       TEXT               -- arbitrary JSON context
    acknowledged    INTEGER  NOT NULL DEFAULT 0
    acked_by        TEXT               -- operator identifier (filled on acknowledge)
    acked_at        TEXT               -- ISO-8601 UTC timestamp of acknowledgement

Usage
-----
::

    from src.privacy.access_control import AlertStore, AccessControlEngine

    store  = AlertStore("data/embeddings.db")
    engine = AccessControlEngine(alert_store=store, on_alert=my_callback)

    # In MQTT subscriber thread:
    engine.on_door_event("door-01", "open", "2026-02-28T10:00:00Z", camera_id="gate-north")

    # After a verify call:
    engine.on_verify_result(
        camera_id="gate-north",
        door_id="door-01",
        subject_key="grp:A:f0:abc",
        matched=False,
        similarity=0.12,
        timestamp_utc="2026-02-28T10:00:01Z",
    )

    # The my_callback(alert_dict) will be invoked synchronously if a rule fires.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Optional

log = logging.getLogger(__name__)

# ─── Constants ─────────────────────────────────────────────────────────────────

FORCED_ENTRY_WINDOW_S: float = 2.0   # ±2 s between door-open and failed verify
TAILGATE_WINDOW_S:     float = 5.0   # sliding window for tailgate detection
BUFFER_TTL_S:          float = 30.0  # max age of entries kept in ring buffers

# ─── DDL ───────────────────────────────────────────────────────────────────────

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS alerts (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_type    TEXT    NOT NULL,
        camera_id     TEXT,
        door_id       TEXT,
        subject_key   TEXT,
        confidence    REAL    NOT NULL DEFAULT 1.0,
        timestamp_utc TEXT    NOT NULL,
        meta_json     TEXT,
        acknowledged  INTEGER NOT NULL DEFAULT 0,
        acked_by      TEXT,
        acked_at      TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_alert_type   ON alerts (alert_type)",
    "CREATE INDEX IF NOT EXISTS idx_alert_door   ON alerts (door_id)",
    "CREATE INDEX IF NOT EXISTS idx_alert_ts     ON alerts (timestamp_utc)",
    "CREATE INDEX IF NOT EXISTS idx_alert_ack    ON alerts (acknowledged)",
]

# Migration: add new columns to existing databases that lack them
_MIGRATIONS = [
    "ALTER TABLE alerts ADD COLUMN acked_by TEXT",
    "ALTER TABLE alerts ADD COLUMN acked_at TEXT",
]

# ─── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class _DoorEvent:
    door_id:       str
    event_type:    str          # "open" | "close"
    timestamp_utc: str
    camera_id:     Optional[str] = None

    @property
    def dt(self) -> datetime:
        return _parse_dt(self.timestamp_utc)


@dataclass
class _VerifyEvent:
    camera_id:     Optional[str]
    door_id:       Optional[str]
    subject_key:   Optional[str]
    matched:       bool
    similarity:    float
    timestamp_utc: str

    @property
    def dt(self) -> datetime:
        return _parse_dt(self.timestamp_utc)


# ─── Helpers ───────────────────────────────────────────────────────────────────


def _parse_dt(ts: str) -> datetime:
    """Parse an ISO-8601 UTC string into a timezone-aware datetime."""
    ts = ts.rstrip("Z")
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        dt = datetime.utcnow()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _utcnow_str() -> str:
    return datetime.now(timezone.utc).isoformat()


def _age_s(ts: str) -> float:
    """Absolute age of a timestamp in seconds (always ≥ 0)."""
    return abs((datetime.now(timezone.utc) - _parse_dt(ts)).total_seconds())


# ─── AlertStore ────────────────────────────────────────────────────────────────


class AlertStore:
    """
    SQLite-backed persistence for security alerts.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file (shared with embedding / attendance
        tables; all coexist safely).
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            for stmt in _DDL:
                con.execute(stmt)
            # Add new columns to existing databases (silently skip if present)
            for stmt in _MIGRATIONS:
                try:
                    con.execute(stmt)
                except sqlite3.OperationalError:
                    pass  # column already exists

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        return con

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        alert_type:    str,
        *,
        camera_id:     Optional[str]   = None,
        door_id:       Optional[str]   = None,
        subject_key:   Optional[str]   = None,
        confidence:    float           = 1.0,
        timestamp_utc: Optional[str]   = None,
        meta:          Optional[dict]  = None,
    ) -> dict[str, Any]:
        """
        Insert a new alert and return it as a dict.

        Parameters
        ----------
        alert_type:
            ``"forced_entry"`` or ``"tailgate"`` (any string accepted for
            extensibility).
        camera_id, door_id, subject_key:
            Optional context.  ``subject_key`` may be a comma-joined list
            when multiple subjects are involved (tailgate).
        confidence:
            Engine-computed confidence score [0, 1].
        timestamp_utc:
            ISO-8601 UTC string; defaults to now.
        meta:
            Arbitrary JSON-serialisable dict for extra context.
        """
        ts   = timestamp_utc or _utcnow_str()
        meta_json = json.dumps(meta) if meta else None
        with self._lock, self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO alerts
                    (alert_type, camera_id, door_id, subject_key, confidence,
                     timestamp_utc, meta_json, acknowledged)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (alert_type, camera_id, door_id, subject_key,
                 confidence, ts, meta_json),
            )
            row_id = cur.lastrowid
        return self.get(row_id)  # type: ignore[return-value]

    def acknowledge(self, alert_id: int, acked_by: Optional[str] = None) -> bool:
        """
        Mark an alert as acknowledged.

        Parameters
        ----------
        alert_id:
            ID of the alert to acknowledge.
        acked_by:
            Optional operator identifier (name, username, session ID) stored
            as an audit trail alongside the acknowledgement timestamp.

        Returns
        -------
        True if the row existed and was updated, False otherwise.
        """
        acked_at = _utcnow_str()
        with self._lock, self._connect() as con:
            cur = con.execute(
                "UPDATE alerts SET acknowledged = 1, acked_by = ?, acked_at = ? WHERE id = ?",
                (acked_by, acked_at, alert_id),
            )
            return cur.rowcount > 0

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, alert_id: int) -> Optional[dict[str, Any]]:
        """Return a single alert dict or None."""
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT * FROM alerts WHERE id = ?", (alert_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_recent(
        self,
        n:              int           = 50,
        alert_type:     Optional[str] = None,
        door_id:        Optional[str] = None,
        unacked_only:   bool          = False,
    ) -> list[dict[str, Any]]:
        """
        Return the most-recent *n* alerts, newest-first.

        Parameters
        ----------
        n:            Maximum rows to return.
        alert_type:   Filter to a specific alert type.
        door_id:      Filter to alerts involving a specific door.
        unacked_only: If True return only un-acknowledged alerts.
        """
        clauses: list[str] = []
        params:  list[Any] = []
        if alert_type:
            clauses.append("alert_type = ?"); params.append(alert_type)
        if door_id:
            clauses.append("door_id = ?"); params.append(door_id)
        if unacked_only:
            clauses.append("acknowledged = 0")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(n)
        with self._lock, self._connect() as con:
            rows = con.execute(
                f"SELECT * FROM alerts {where} ORDER BY id DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_unacked(self) -> int:
        """Return the number of unacknowledged alerts."""
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT COUNT(*) FROM alerts WHERE acknowledged = 0"
            ).fetchone()
        return int(row[0]) if row else 0

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        if d.get("meta_json"):
            try:
                d["meta"] = json.loads(d["meta_json"])
            except Exception:
                d["meta"] = {}
        else:
            d["meta"] = {}
        d.pop("meta_json", None)
        d["acknowledged"] = bool(d.get("acknowledged", 0))
        # Ensure audit fields are always present (None for un-acked rows)
        d.setdefault("acked_by", None)
        d.setdefault("acked_at", None)
        return d


# ─── AccessControlEngine ───────────────────────────────────────────────────────


class AccessControlEngine:
    """
    Thread-safe access-control correlation engine.

    Parameters
    ----------
    alert_store:
        :class:`AlertStore` instance used to persist generated alerts.
    on_alert:
        Optional callback invoked **synchronously** (within the calling thread)
        whenever a new alert is generated.  Signature::

            def on_alert(alert: dict) -> None: ...

        Use this to broadcast the alert over Socket.IO, post a webhook, etc.

    forced_entry_window_s:
        Time window (seconds) for forced-entry correlation (default 2.0).
    tailgate_window_s:
        Sliding time window (seconds) for tailgate detection (default 5.0).
    buffer_ttl_s:
        Entries older than this are pruned from the ring buffers (default 30.0).
    """

    def __init__(
        self,
        alert_store:            AlertStore,
        on_alert:               Optional[Callable[[dict], None]] = None,
        forced_entry_window_s:  float = FORCED_ENTRY_WINDOW_S,
        tailgate_window_s:      float = TAILGATE_WINDOW_S,
        buffer_ttl_s:           float = BUFFER_TTL_S,
    ) -> None:
        self._store                 = alert_store
        self._on_alert              = on_alert
        self._forced_window         = forced_entry_window_s
        self._tailgate_window       = tailgate_window_s
        self._buffer_ttl            = buffer_ttl_s
        self._lock                  = threading.Lock()

        # door_id → deque of _DoorEvent
        self._door_buf: dict[str, Deque[_DoorEvent]]   = {}
        # door_id → deque of _VerifyEvent
        self._verify_buf: dict[str, Deque[_VerifyEvent]] = {}
        # Set of (door_id, open_ts_str) pairs already used in a forced_entry alert
        # prevents duplicate alerts for the same door-open event
        self._used_door_opens: set[tuple[str, str]] = set()
        # Set of frozensets of subject_keys already used in a tailgate alert within window
        self._used_tailgate: set[str] = set()

    # ── Public API ────────────────────────────────────────────────────────────

    def on_door_event(
        self,
        door_id:       str,
        event_type:    str,
        timestamp_utc: str,
        camera_id:     Optional[str] = None,
    ) -> Optional[dict]:
        """
        Record a door sensor event and check for forced-entry.

        Parameters
        ----------
        door_id:       Door that fired the event (matches ``door_id`` in cameras.yaml).
        event_type:    ``"open"`` or ``"close"``.
        timestamp_utc: ISO-8601 UTC timestamp from the MQTT payload.
        camera_id:     Camera associated with this door (optional, for richer alerts).

        Returns
        -------
        Alert dict if a forced-entry alert was generated, else ``None``.
        """
        ev = _DoorEvent(
            door_id=door_id,
            event_type=event_type,
            timestamp_utc=timestamp_utc,
            camera_id=camera_id,
        )
        with self._lock:
            self._door_buf.setdefault(door_id, deque())
            self._door_buf[door_id].append(ev)
            self._prune(self._door_buf[door_id], ev.dt)

            alert = None
            if event_type == "open":
                alert = self._check_forced_entry_on_door_open(ev)
        if alert:
            self._fire(alert)
        return alert

    def on_verify_result(
        self,
        camera_id:     Optional[str],
        door_id:       Optional[str],
        subject_key:   Optional[str],
        matched:       bool,
        similarity:    float,
        timestamp_utc: Optional[str] = None,
    ) -> list[dict]:
        """
        Record a verify result and check for forced-entry + tailgate.

        Parameters
        ----------
        camera_id:     Camera that captured the face (optional).
        door_id:       Door associated with this camera (optional).
        subject_key:   The opaque identity key (or None for anonymous).
        matched:       Whether the verify succeeded.
        similarity:    Cosine similarity score.
        timestamp_utc: ISO-8601 UTC timestamp; defaults to now().

        Returns
        -------
        List of alert dicts generated (may be empty, one, or two entries).
        """
        ts = timestamp_utc or _utcnow_str()
        ev = _VerifyEvent(
            camera_id=camera_id,
            door_id=door_id,
            subject_key=subject_key,
            matched=matched,
            similarity=similarity,
            timestamp_utc=ts,
        )
        alerts: list[dict] = []
        with self._lock:
            if door_id:
                self._verify_buf.setdefault(door_id, deque())
                self._verify_buf[door_id].append(ev)
                self._prune(self._verify_buf[door_id], ev.dt)

                if not matched:
                    a = self._check_forced_entry_on_verify(ev)
                    if a:
                        alerts.append(a)

                a2 = self._check_tailgate(ev)
                if a2:
                    alerts.append(a2)

        for a in alerts:
            self._fire(a)
        return alerts

    # ── Forced-entry checks ───────────────────────────────────────────────────

    def _check_forced_entry_on_door_open(self, door_ev: _DoorEvent) -> Optional[dict]:
        """
        Called when a door OPEN event is received.

        Scans the verify buffer for the same door for a recent FAILED verify
        within the ±forced_window seconds.
        """
        buf = self._verify_buf.get(door_ev.door_id, deque())
        for vev in buf:
            if vev.matched:
                continue
            diff = abs((door_ev.dt - vev.dt).total_seconds())
            if diff <= self._forced_window:
                key = (door_ev.door_id, door_ev.timestamp_utc)
                if key in self._used_door_opens:
                    continue
                self._used_door_opens.add(key)
                return self._store.add(
                    "forced_entry",
                    camera_id=door_ev.camera_id or vev.camera_id,
                    door_id=door_ev.door_id,
                    subject_key=vev.subject_key,
                    confidence=round(1.0 - max(vev.similarity, 0.0), 4),
                    timestamp_utc=door_ev.timestamp_utc,
                    meta={
                        "trigger":        "door_open_with_failed_verify",
                        "door_event_ts":  door_ev.timestamp_utc,
                        "verify_event_ts": vev.timestamp_utc,
                        "time_diff_s":    round(diff, 3),
                        "similarity":     round(vev.similarity, 4),
                    },
                )
        return None

    def _check_forced_entry_on_verify(self, vev: _VerifyEvent) -> Optional[dict]:
        """
        Called when a FAILED verify result arrives.

        Scans the door buffer for the same door for a recent OPEN event
        within the ±forced_window seconds.
        """
        if not vev.door_id:
            return None
        buf = self._door_buf.get(vev.door_id, deque())
        for dev in buf:
            if dev.event_type != "open":
                continue
            diff = abs((vev.dt - dev.dt).total_seconds())
            if diff <= self._forced_window:
                key = (vev.door_id, dev.timestamp_utc)
                if key in self._used_door_opens:
                    continue
                self._used_door_opens.add(key)
                return self._store.add(
                    "forced_entry",
                    camera_id=vev.camera_id or dev.camera_id,
                    door_id=vev.door_id,
                    subject_key=vev.subject_key,
                    confidence=round(1.0 - max(vev.similarity, 0.0), 4),
                    timestamp_utc=vev.timestamp_utc,
                    meta={
                        "trigger":         "failed_verify_with_door_open",
                        "door_event_ts":   dev.timestamp_utc,
                        "verify_event_ts": vev.timestamp_utc,
                        "time_diff_s":     round(diff, 3),
                        "similarity":      round(vev.similarity, 4),
                    },
                )
        return None

    # ── Tailgate check ────────────────────────────────────────────────────────

    def _check_tailgate(self, new_ev: _VerifyEvent) -> Optional[dict]:
        """
        Called after every verify result for a door with an associated door_id.

        Gathers all verify events for the same door within the last
        tailgate_window_s seconds.  A tailgate alert fires when:
          • ≥ 2 distinct subjects are in the window, AND
          • at least one of them is unmatched.
        """
        if not new_ev.door_id:
            return None
        buf = self._verify_buf.get(new_ev.door_id, deque())
        cutoff_s = self._tailgate_window
        window: list[_VerifyEvent] = [
            e for e in buf
            if abs((new_ev.dt - e.dt).total_seconds()) <= cutoff_s
        ]
        if len(window) < 2:
            return None
        has_unmatched = any(not e.matched for e in window)
        if not has_unmatched:
            return None

        keys = frozenset(e.subject_key or "" for e in window)
        cache_key = f"{new_ev.door_id}:{','.join(sorted(keys))}"
        if cache_key in self._used_tailgate:
            return None
        self._used_tailgate.add(cache_key)

        unmatched_keys = [e.subject_key for e in window if not e.matched]
        matched_keys   = [e.subject_key for e in window if e.matched]
        return self._store.add(
            "tailgate",
            camera_id=new_ev.camera_id,
            door_id=new_ev.door_id,
            subject_key=",".join(k for k in (e.subject_key or "" for e in window) if k),
            confidence=1.0,
            timestamp_utc=new_ev.timestamp_utc,
            meta={
                "trigger":       "tailgate_detected",
                "n_persons":     len(window),
                "matched_keys":  matched_keys,
                "unmatched_keys": unmatched_keys,
                "window_s":      cutoff_s,
            },
        )

    # ── Buffer maintenance ────────────────────────────────────────────────────

    def _prune(self, buf: deque, reference_dt: datetime) -> None:
        """Remove entries older than buffer_ttl_s relative to ``reference_dt``."""
        while buf:
            age = abs((reference_dt - buf[0].dt).total_seconds())
            if age > self._buffer_ttl:
                buf.popleft()
            else:
                break

    # ── Fire callback ─────────────────────────────────────────────────────────

    def _fire(self, alert: dict) -> None:
        """Invoke the on_alert callback (if set) with the alert dict."""
        if self._on_alert:
            try:
                self._on_alert(alert)
            except Exception:
                log.exception("[ACE] on_alert callback raised an exception")
