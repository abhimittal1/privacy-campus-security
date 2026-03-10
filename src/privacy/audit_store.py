"""
audit_store.py – Immutable audit log for Privacy Campus Security.

Design
------
* Append-only SQLite table.  Rows are never updated or deleted.
* A DB-level TRIGGER prevents any UPDATE or DELETE on the ``audit_log`` table
  so the record cannot be silently tampered with from within the application.
* Each row captures: timestamp, event type, actor (username from JWT or
  ``"system"``), optional subject/camera/alert identifiers, and a free-form
  JSON ``detail`` blob.

Event types
-----------
enroll          – single-subject enrollment
enroll_batch    – multi-face batch enrollment
verify          – single-subject 1-to-1 verification
verify_frame    – multi-face frame verification (logs n_faces / n_matched)
ack             – alert acknowledged (includes alert_id, alert_type, acked_by)
panic           – panic escalation triggered
alert           – new access-control alert persisted (logged by the engine)

Environment variables
---------------------
PCS_AUDIT_DB    Path to the SQLite audit database.
                Default: same directory as PCS_STORE → ``data/audit.db``
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_utc TEXT    NOT NULL,
    event_type    TEXT    NOT NULL,
    actor         TEXT,
    subject_id    TEXT,
    camera_id     TEXT,
    alert_id      INTEGER,
    detail        TEXT
);

-- Prevent any UPDATE on audit rows
CREATE TRIGGER IF NOT EXISTS audit_no_update
    BEFORE UPDATE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log rows are immutable');
END;

-- Prevent any DELETE on audit rows
CREATE TRIGGER IF NOT EXISTS audit_no_delete
    BEFORE DELETE ON audit_log
BEGIN
    SELECT RAISE(ABORT, 'audit_log rows are immutable');
END;
"""


class AuditStore:
    """
    Append-only audit log backed by SQLite.

    Args:
        db_path: Path to the SQLite file.  Parent dirs are created
                 automatically.  Falls back to ``PCS_AUDIT_DB`` env var or
                 ``data/audit.db``.
    """

    def __init__(self, db_path: Optional[str | Path] = None) -> None:
        if db_path is None:
            db_path = os.environ.get("PCS_AUDIT_DB", "data/audit.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()
        log.debug("AuditStore initialised at %s", self.db_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(
        self,
        event_type: str,
        *,
        actor: Optional[str] = None,
        subject_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        alert_id: Optional[int] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Append one immutable audit record.

        Args:
            event_type: One of the documented event types (enroll, verify,
                        ack, panic, alert, …).
            actor:      Username of the authenticated caller or ``"system"``.
            subject_id: Subject key / ID involved (optional).
            camera_id:  Camera involved (optional).
            alert_id:   Related alert row ID (optional).
            detail:     Extra structured data serialised to JSON.

        Returns:
            The ``id`` of the newly inserted row.
        """
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        detail_json = json.dumps(detail, default=str) if detail else None
        cur = self._conn.execute(
            """
            INSERT INTO audit_log
                (timestamp_utc, event_type, actor, subject_id, camera_id, alert_id, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (now, event_type, actor, subject_id, camera_id, alert_id, detail_json),
        )
        self._conn.commit()
        row_id = cur.lastrowid
        log.debug("audit: %s actor=%r row_id=%s", event_type, actor, row_id)
        return row_id  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        *,
        limit: int = 100,
        event_type: Optional[str] = None,
        actor: Optional[str] = None,
        camera_id: Optional[str] = None,
        alert_id: Optional[int] = None,
        since: Optional[str] = None,
    ) -> List[Dict]:
        """
        Return audit rows, newest first.

        Args:
            limit:      Maximum number of rows to return (default 100, max 1000).
            event_type: Filter by exact event_type string.
            actor:      Filter by exact actor username.
            camera_id:  Filter by camera_id.
            alert_id:   Filter by alert_id.
            since:      ISO-8601 timestamp — return only rows after this point.

        Returns:
            List of dicts with keys: id, timestamp_utc, event_type, actor,
            subject_id, camera_id, alert_id, detail (parsed JSON or None).
        """
        limit = min(int(limit), 1000)

        conditions: List[str] = []
        params:     List[Any] = []

        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type)
        if actor:
            conditions.append("actor = ?")
            params.append(actor)
        if camera_id:
            conditions.append("camera_id = ?")
            params.append(camera_id)
        if alert_id is not None:
            conditions.append("alert_id = ?")
            params.append(alert_id)
        if since:
            conditions.append("timestamp_utc > ?")
            params.append(since)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)

        rows = self._conn.execute(
            f"SELECT * FROM audit_log {where} ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()

        result = []
        for r in rows:
            d = dict(r)
            if d["detail"]:
                try:
                    d["detail"] = json.loads(d["detail"])
                except json.JSONDecodeError:
                    pass  # leave as raw string
            result.append(d)
        return result

    def count(self) -> int:
        """Total number of audit rows."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM audit_log"
        ).fetchone()[0]

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()
