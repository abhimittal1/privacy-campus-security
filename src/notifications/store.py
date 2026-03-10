"""
notifications/store.py – SQLite-backed persistence for notification records.

Schema (``notifications`` table)
---------------------------------
::

    id            INTEGER  PRIMARY KEY AUTOINCREMENT
    alert_id      INTEGER               – FK to alerts.id (nullable for non-alert events)
    alert_type    TEXT     NOT NULL     – e.g. forced_entry, safety_alert, loitering
    severity      TEXT     NOT NULL     – high | medium | low
    channel       TEXT     NOT NULL     – websocket | smtp | sms
    status        TEXT     NOT NULL DEFAULT 'pending'
                                        – pending | sent | failed | throttled
    attempts      INTEGER  NOT NULL DEFAULT 0
    last_attempt  TEXT                  – ISO-8601 UTC of last delivery attempt
    next_retry    TEXT                  – ISO-8601 UTC before which no retry should happen
    error_msg     TEXT                  – last error message on failure
    payload_json  TEXT                  – JSON-serialised notification payload
    created_at    TEXT     NOT NULL     – ISO-8601 UTC of record creation
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
    CREATE TABLE IF NOT EXISTS notifications (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_id      INTEGER,
        alert_type    TEXT    NOT NULL,
        severity      TEXT    NOT NULL,
        channel       TEXT    NOT NULL,
        status        TEXT    NOT NULL DEFAULT 'pending',
        attempts      INTEGER NOT NULL DEFAULT 0,
        last_attempt  TEXT,
        next_retry    TEXT,
        error_msg     TEXT,
        payload_json  TEXT,
        created_at    TEXT    NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_notif_status   ON notifications (status)",
    "CREATE INDEX IF NOT EXISTS idx_notif_channel  ON notifications (channel)",
    "CREATE INDEX IF NOT EXISTS idx_notif_alert_id ON notifications (alert_id)",
    "CREATE INDEX IF NOT EXISTS idx_notif_created  ON notifications (created_at)",
]


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    if d.get("payload_json"):
        try:
            d["payload"] = json.loads(d["payload_json"])
        except Exception:
            d["payload"] = {}
    else:
        d["payload"] = {}
    d.pop("payload_json", None)
    return d


class NotificationStore:
    """
    Thread-safe SQLite store for notification delivery records.

    All operations use a per-instance lock so multiple threads (Flask workers,
    retry daemon, MQTT subscriber) can call concurrently without corruption.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  The ``notifications`` table coexists
        safely alongside the ``alerts``, ``attendance``, and other tables.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    # ── internal ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._path, check_same_thread=False, timeout=15)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_db(self) -> None:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._connect() as con:
            for stmt in _DDL:
                con.execute(stmt)

    # ── write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        *,
        alert_id:   Optional[int]  = None,
        alert_type: str,
        severity:   str,
        channel:    str,
        payload:    Optional[dict] = None,
        status:     str            = "pending",
    ) -> dict[str, Any]:
        """
        Insert a new notification record with *status* ``'pending'`` by default.

        Returns the inserted row as a dict (``payload_json`` replaced by ``payload``).
        """
        ts           = _utcnow()
        payload_json = json.dumps(payload) if payload is not None else None
        with self._lock, self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO notifications
                    (alert_id, alert_type, severity, channel, status,
                     attempts, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (alert_id, alert_type, severity, channel, status, payload_json, ts),
            )
            row_id = cur.lastrowid
        return self.get(row_id)  # type: ignore[return-value]

    def update_status(
        self,
        notification_id: int,
        status:          str,
        *,
        error_msg:  Optional[str] = None,
        next_retry: Optional[str] = None,
    ) -> None:
        """Update the delivery status (and optional error / retry timestamp)."""
        ts = _utcnow()
        with self._lock, self._connect() as con:
            con.execute(
                """
                UPDATE notifications
                SET status = ?, last_attempt = ?, error_msg = ?, next_retry = ?
                WHERE id = ?
                """,
                (status, ts, error_msg, next_retry, notification_id),
            )

    def increment_attempts(self, notification_id: int) -> None:
        """Atomically increment the attempt counter."""
        with self._lock, self._connect() as con:
            con.execute(
                "UPDATE notifications SET attempts = attempts + 1 WHERE id = ?",
                (notification_id,),
            )

    # ── read ──────────────────────────────────────────────────────────────────

    def get(self, notification_id: int) -> Optional[dict[str, Any]]:
        """Return a single notification dict or None."""
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT * FROM notifications WHERE id = ?", (notification_id,)
            ).fetchone()
        return _row_to_dict(row) if row else None

    def get_pending_retry(self, *, max_retries: int = 3) -> list[dict[str, Any]]:
        """
        Return failed notifications that are due for a retry.

        Criteria
        --------
        * ``status = 'failed'``
        * ``attempts < max_retries``
        * ``next_retry IS NULL  OR  next_retry <= utcnow``
        """
        now = _utcnow()
        with self._lock, self._connect() as con:
            rows = con.execute(
                """
                SELECT * FROM notifications
                WHERE  status   = 'failed'
                AND    attempts < ?
                AND    (next_retry IS NULL OR next_retry <= ?)
                ORDER  BY id ASC
                """,
                (max_retries, now),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def list_recent(
        self,
        n:          int           = 50,
        *,
        channel:    Optional[str] = None,
        status:     Optional[str] = None,
        alert_id:   Optional[int] = None,
        alert_type: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Return the most recent *n* notification records, newest-first.

        All filter parameters are optional and combinable.
        """
        clauses: list[str] = []
        params:  list[Any] = []

        if channel:
            clauses.append("channel = ?");    params.append(channel)
        if status:
            clauses.append("status = ?");     params.append(status)
        if alert_id is not None:
            clauses.append("alert_id = ?");   params.append(alert_id)
        if alert_type:
            clauses.append("alert_type = ?"); params.append(alert_type)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(n)

        with self._lock, self._connect() as con:
            rows = con.execute(
                f"SELECT * FROM notifications {where} ORDER BY id DESC LIMIT ?",
                params,
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def count_by_status(self, status: str) -> int:
        """Return the number of notifications with the given status."""
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT COUNT(*) FROM notifications WHERE status = ?", (status,)
            ).fetchone()
        return int(row[0]) if row else 0
