"""
loitering_store.py – SQLite-backed persistence for loitering detection events.

Schema (``loitering_events`` table)
-------------------------------------
::

    id              INTEGER  PRIMARY KEY AUTOINCREMENT
    camera_id       TEXT     NOT NULL               – which camera detected the loitering
    subject_key     TEXT     NOT NULL               – opaque hashed subject identifier
    first_seen_utc  TEXT     NOT NULL               – ISO-8601 UTC when subject first appeared
    detected_utc    TEXT     NOT NULL               – ISO-8601 UTC when dwell threshold was crossed
    dwell_s         REAL     NOT NULL DEFAULT 0     – seconds subject dwelled before triggering
    dismissed       INTEGER  NOT NULL DEFAULT 0     – 1 → operator dismissed the event
    action_by       TEXT                            – operator who acted
    action_at       TEXT                            – ISO-8601 UTC of the action

Usage
-----
::

    from src.privacy.loitering_store import LoiteringStore

    store = LoiteringStore("data/attendance.db")
    ev    = store.add("gate-north", "grp:A:f0:abc", dwell_s=65.0)
    store.dismiss(ev["id"], action_by="Officer Kim")
    events = store.list_recent(n=50, dismissed=False)
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


_DDL = [
    """
    CREATE TABLE IF NOT EXISTS loitering_events (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id      TEXT    NOT NULL,
        subject_key    TEXT    NOT NULL,
        first_seen_utc TEXT    NOT NULL,
        detected_utc   TEXT    NOT NULL,
        dwell_s        REAL    NOT NULL DEFAULT 0,
        dismissed      INTEGER NOT NULL DEFAULT 0,
        action_by      TEXT,
        action_at      TEXT
    )
    """,
]


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["dismissed"] = bool(d.get("dismissed", 0))
    return d


class LoiteringStore:
    """
    Thread-safe SQLite store for loitering detection events.

    All public methods acquire a per-instance lock before touching the DB,
    so multiple threads (Flask request workers, MQTT subscriber) can call
    concurrently without corruption.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    # ── internal ─────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            with self._connect() as conn:
                for ddl in _DDL:
                    conn.execute(ddl)

    # ── public API ────────────────────────────────────────────────────────────

    def add(
        self,
        camera_id: str,
        subject_key: str,
        *,
        first_seen_utc: str | None = None,
        dwell_s: float = 0.0,
    ) -> dict:
        """
        Persist a new loitering event and return it as a dict.

        Parameters
        ----------
        camera_id       : which camera triggered the event
        subject_key     : opaque hashed face identifier from the embedding store
        first_seen_utc  : ISO-8601 UTC string when the subject first appeared;
                          defaults to now if omitted
        dwell_s         : seconds the subject remained before the threshold was crossed
        """
        now = _utcnow()
        first_seen_utc = first_seen_utc or now

        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO loitering_events
                        (camera_id, subject_key, first_seen_utc, detected_utc, dwell_s)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (camera_id, subject_key, first_seen_utc, now, dwell_s),
                )
                event_id = cursor.lastrowid
                row = conn.execute(
                    "SELECT * FROM loitering_events WHERE id = ?", (event_id,)
                ).fetchone()
        return _row_to_dict(row)

    def dismiss(self, event_id: int, *, action_by: str | None = None) -> bool:
        """
        Mark event as dismissed.

        Returns True if the row existed and was updated, False otherwise.
        """
        now = _utcnow()
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    UPDATE loitering_events
                    SET dismissed=1, action_by=?, action_at=?
                    WHERE id=?
                    """,
                    (action_by, now, event_id),
                ).rowcount
        return rows > 0

    def get(self, event_id: int) -> dict | None:
        """Return a single event dict, or None if not found."""
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM loitering_events WHERE id=?", (event_id,)
                ).fetchone()
        return _row_to_dict(row) if row else None

    def list_recent(
        self,
        n: int = 50,
        *,
        camera_id: str | None = None,
        subject_key: str | None = None,
        dismissed: bool | None = None,
    ) -> list[dict]:
        """
        Return up to *n* events newest-first.

        Filters
        -------
        camera_id   : restrict to this camera
        subject_key : restrict to this subject
        dismissed   : if True → only dismissed; if False → only active; None → all
        """
        clauses: list[str] = []
        params: list[Any] = []

        if camera_id is not None:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if subject_key is not None:
            clauses.append("subject_key = ?")
            params.append(subject_key)
        if dismissed is True:
            clauses.append("dismissed = 1")
        elif dismissed is False:
            clauses.append("dismissed = 0")

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM loitering_events {where} ORDER BY id DESC LIMIT ?"
        params.append(n)

        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]

    def count_active(self) -> int:
        """Return the number of undismissed loitering events."""
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM loitering_events WHERE dismissed=0"
                ).fetchone()
        return int(row[0])
