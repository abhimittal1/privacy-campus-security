"""
left_object_store.py – SQLite-backed persistence for left-object detection events.

Schema (``left_object_events`` table)
--------------------------------------
::

    id              INTEGER  PRIMARY KEY AUTOINCREMENT
    camera_id       TEXT     NOT NULL               – which camera detected the object
    bbox_json       TEXT                            – JSON [x, y, w, h] in original-frame pixels
    first_seen_utc  TEXT     NOT NULL               – ISO-8601 UTC when the object first appeared
    detected_utc    TEXT     NOT NULL               – ISO-8601 UTC when dwell threshold was crossed
    dwell_s         REAL     NOT NULL DEFAULT 0     – seconds the object dwelled before triggering
    snapshot_b64    TEXT                            – data-URI JPEG crop of the region (nullable)
    dismissed       INTEGER  NOT NULL DEFAULT 0     – 1 → operator marked it "removed"
    confirmed       INTEGER  NOT NULL DEFAULT 0     – 1 → operator confirmed it as a security threat
    action_by       TEXT                            – operator name who acted
    action_at       TEXT                            – ISO-8601 UTC of the action

Usage
-----
::

    from src.privacy.left_object_store import LeftObjectStore

    store = LeftObjectStore("data/attendance.db")
    ev    = store.add("gate-north", bbox=[120, 80, 60, 90], dwell_s=31.2, ...)
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
    CREATE TABLE IF NOT EXISTS left_object_events (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id      TEXT    NOT NULL,
        bbox_json      TEXT,
        first_seen_utc TEXT    NOT NULL,
        detected_utc   TEXT    NOT NULL,
        dwell_s        REAL    NOT NULL DEFAULT 0,
        snapshot_b64   TEXT,
        dismissed      INTEGER NOT NULL DEFAULT 0,
        confirmed      INTEGER NOT NULL DEFAULT 0,
        action_by      TEXT,
        action_at      TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_lo_camera ON left_object_events (camera_id)",
    "CREATE INDEX IF NOT EXISTS idx_lo_dets   ON left_object_events (detected_utc)",
    "CREATE INDEX IF NOT EXISTS idx_lo_dis    ON left_object_events (dismissed)",
]


class LeftObjectStore:
    """Thread-safe SQLite store for left-object detection events."""

    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        self._lock = threading.Lock()
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with self._con() as con:
            for stmt in _DDL:
                con.execute(stmt)

    def _con(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._path, check_same_thread=False)
        con.row_factory = sqlite3.Row
        return con

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        try:
            d["bbox"] = json.loads(d.pop("bbox_json") or "null")
        except Exception:
            d["bbox"] = None
        return d

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        camera_id: str,
        *,
        bbox: Optional[list[int]] = None,
        first_seen_utc: Optional[str] = None,
        dwell_s: float = 0.0,
        snapshot_b64: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Persist a new left-object event and return it as a dict.

        Parameters
        ----------
        camera_id:
            The camera that observed the object.
        bbox:
            Bounding box ``[x, y, w, h]`` in original-frame pixel coordinates.
        first_seen_utc:
            ISO-8601 UTC when the object first appeared (defaults to now).
        dwell_s:
            Seconds the object had been stationary before the threshold was crossed.
        snapshot_b64:
            Optional base64 data-URI JPEG crop of the region.
        """
        now = _utcnow()
        first = first_seen_utc or now
        bbox_json = json.dumps(bbox) if bbox is not None else None
        with self._lock, self._con() as con:
            cur = con.execute(
                """
                INSERT INTO left_object_events
                    (camera_id, bbox_json, first_seen_utc, detected_utc,
                     dwell_s, snapshot_b64)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (camera_id, bbox_json, first, now, dwell_s, snapshot_b64),
            )
            row_id = cur.lastrowid
        return self.get(row_id)  # type: ignore[return-value]

    def _set_action(
        self,
        event_id: int,
        *,
        dismissed: Optional[int] = None,
        confirmed: Optional[int] = None,
        action_by: Optional[str] = None,
    ) -> bool:
        parts, vals = [], []
        if dismissed is not None:
            parts.append("dismissed = ?"); vals.append(dismissed)
        if confirmed is not None:
            parts.append("confirmed = ?"); vals.append(confirmed)
        parts.append("action_at = ?"); vals.append(_utcnow())
        parts.append("action_by = ?"); vals.append(action_by or None)
        vals.append(event_id)
        with self._lock, self._con() as con:
            cur = con.execute(
                f"UPDATE left_object_events SET {', '.join(parts)} WHERE id = ?", vals
            )
        return cur.rowcount > 0

    def dismiss(self, event_id: int, *, action_by: Optional[str] = None) -> bool:
        """Mark the object as removed / no longer present."""
        return self._set_action(event_id, dismissed=1, action_by=action_by)

    def confirm(self, event_id: int, *, action_by: Optional[str] = None) -> bool:
        """Confirm the object as a security threat (unattended bag, etc)."""
        return self._set_action(event_id, confirmed=1, action_by=action_by)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, event_id: int) -> Optional[dict[str, Any]]:
        with self._con() as con:
            row = con.execute(
                "SELECT * FROM left_object_events WHERE id = ?", (event_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_recent(
        self,
        n: int = 50,
        *,
        camera_id: Optional[str] = None,
        dismissed: Optional[bool] = None,
        confirmed: Optional[bool] = None,
    ) -> list[dict[str, Any]]:
        """
        Return up to *n* events, newest first.

        Parameters
        ----------
        camera_id:
            Filter to one camera; ``None`` returns all cameras.
        dismissed / confirmed:
            ``True`` → only that action; ``False`` → exclude that action;
            ``None`` → no filter.
        """
        where, vals = [], []
        if camera_id:
            where.append("camera_id = ?"); vals.append(camera_id)
        if dismissed is not None:
            where.append("dismissed = ?"); vals.append(int(dismissed))
        if confirmed is not None:
            where.append("confirmed = ?"); vals.append(int(confirmed))
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        vals.append(n)
        with self._con() as con:
            rows = con.execute(
                f"SELECT * FROM left_object_events {clause} ORDER BY id DESC LIMIT ?",
                vals,
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_active(self) -> int:
        """Return the number of events that are neither dismissed nor confirmed."""
        with self._con() as con:
            return con.execute(
                "SELECT COUNT(*) FROM left_object_events WHERE dismissed=0 AND confirmed=0"
            ).fetchone()[0]
