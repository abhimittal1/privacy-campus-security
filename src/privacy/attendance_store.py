"""
attendance_store.py – Session-aware attendance log backed by SQLite.

Schema
------
The ``attendance`` table is created in whichever SQLite file you point it at.
When ``db_path`` is the same file used by :class:`~src.privacy.embedding_store.EmbeddingStore`
(the default), the two tables coexist safely — SQLite supports multiple tables
per file and the GIL prevents concurrent-writer data races.

Table columns
~~~~~~~~~~~~~
::

    id            INTEGER  PRIMARY KEY AUTOINCREMENT
    subject_key   TEXT     NOT NULL           – opaque / hashed subject key
    group_id      TEXT     NOT NULL DEFAULT ''
    camera_id     TEXT     NOT NULL DEFAULT 'default'
    timestamp_utc TEXT     NOT NULL           – ISO-8601 UTC string
    similarity    REAL     NOT NULL           – cosine similarity score at mark time
    matched       INTEGER  NOT NULL           – 1 = matched, 0 = not matched

Indices are created on ``subject_key``, ``group_id``, ``camera_id``, and
``timestamp_utc`` to keep range queries fast even for large attendance logs.

Session TTL (deduplication)
---------------------------
A real-time camera posts a frame every N milliseconds, which would flood the
table with thousands of rows for a single face.  To prevent this, each
:meth:`AttendanceStore.mark` call checks whether the same
``(subject_key, camera_id)`` pair has a row whose ``timestamp_utc`` already
falls within the last ``session_ttl_s`` seconds.  If so, the new row is
silently **skipped** and the method returns ``False``.

Default TTL: **300 s (5 min)**.  Override via the ``PCS_SESSION_TTL``
environment variable.

Usage example
-------------
::

    from src.privacy.attendance_store import AttendanceStore

    att = AttendanceStore("data/embeddings.db")
    att.mark(subject_key="abc123", similarity=0.87, matched=True,
             camera_id="entrance-cam", group_id="SHIFT-A")

    report = att.query_aggregated(from_ts="2026-02-28T00:00:00Z",
                                  group_id="SHIFT-A")
    print(report["per_subject"])
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────────

_DEFAULT_TTL_S: int = int(os.environ.get("PCS_SESSION_TTL", "300"))

# ─── DDL ───────────────────────────────────────────────────────────────────────

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS attendance (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_key   TEXT    NOT NULL,
        group_id      TEXT    NOT NULL DEFAULT '',
        camera_id     TEXT    NOT NULL DEFAULT 'default',
        timestamp_utc TEXT    NOT NULL,
        similarity    REAL    NOT NULL,
        matched       INTEGER NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_att_subject   ON attendance (subject_key)",
    "CREATE INDEX IF NOT EXISTS idx_att_group     ON attendance (group_id)",
    "CREATE INDEX IF NOT EXISTS idx_att_camera    ON attendance (camera_id)",
    "CREATE INDEX IF NOT EXISTS idx_att_ts        ON attendance (timestamp_utc)",
]


# ─── Public class ──────────────────────────────────────────────────────────────


class AttendanceStore:
    """
    Thin SQLite wrapper for the campus-security attendance log.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  If it is the *same* file as the
        :class:`~src.privacy.embedding_store.EmbeddingStore`'s SQLite file,
        the ``attendance`` table is safely added alongside ``embeddings``
        without touching existing data.
    session_ttl_s:
        Seconds within which a duplicate ``(subject_key, camera_id)`` mark is
        silently dropped.  Defaults to the ``PCS_SESSION_TTL`` env var, then
        300 (5 min).
    """

    def __init__(
        self,
        db_path: str | Path,
        session_ttl_s: int = _DEFAULT_TTL_S,
    ) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._ttl  = int(session_ttl_s)

        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        for stmt in _DDL:
            self._conn.execute(stmt)
        self._conn.commit()

        log.debug(
            "AttendanceStore initialised at %s  ttl=%ds", self._path, self._ttl
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Write
    # ──────────────────────────────────────────────────────────────────────────

    def mark(
        self,
        *,
        subject_key: str,
        similarity: float,
        matched: bool,
        group_id: str = "",
        camera_id: str = "default",
        timestamp_utc: str | None = None,
    ) -> bool:
        """
        Insert one attendance record, respecting the session TTL.

        Parameters
        ----------
        subject_key:   Opaque subject identifier (e.g. a hashed store key or
                       the raw plaintext key when privacy is handled upstream).
        similarity:    Cosine similarity score from the verification step.
        matched:       Whether the verification was a positive match.
        group_id:      Optional group tag (e.g. ``"SHIFT-A"``).
        camera_id:     Camera / sensor identifier (e.g. ``"entrance-cam"``).
        timestamp_utc: ISO-8601 UTC timestamp.  Defaults to *now* if omitted.

        Returns
        -------
        ``True``  – row inserted successfully.
        ``False`` – duplicate within the TTL window; row skipped.
        """
        if not subject_key:
            raise ValueError("subject_key must not be empty")

        now = timestamp_utc or datetime.now(timezone.utc).isoformat()

        # ── TTL dedup ──────────────────────────────────────────────────────────
        cur = self._conn.execute(
            """
            SELECT MAX(timestamp_utc) AS last_ts
            FROM   attendance
            WHERE  subject_key = ? AND camera_id = ?
            """,
            (subject_key, camera_id),
        )
        row = cur.fetchone()
        if row and row["last_ts"]:
            try:
                last_dt = datetime.fromisoformat(
                    row["last_ts"].replace("Z", "+00:00")
                )
                now_dt = datetime.fromisoformat(now.replace("Z", "+00:00"))
                if (now_dt - last_dt).total_seconds() < self._ttl:
                    log.debug(
                        "TTL guard: skipping duplicate mark for %s @ %s",
                        subject_key,
                        camera_id,
                    )
                    return False
            except ValueError:
                pass  # unparseable timestamp → allow insert

        self._conn.execute(
            """
            INSERT INTO attendance
                (subject_key, group_id, camera_id, timestamp_utc, similarity, matched)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (subject_key, group_id, camera_id, now, float(similarity), int(matched)),
        )
        self._conn.commit()
        log.debug(
            "Attendance marked: subject=%s  cam=%s  sim=%.4f  match=%s",
            subject_key,
            camera_id,
            similarity,
            matched,
        )
        return True

    def mark_many(
        self, records: list[dict[str, Any]]
    ) -> tuple[int, int]:
        """
        Mark multiple attendance records, applying TTL deduplication to each.

        Parameters
        ----------
        records:
            List of dicts.  Each dict accepts the same keys as :meth:`mark`:
            ``subject_key`` (required), ``similarity`` (required),
            ``matched`` (required), ``group_id``, ``camera_id``,
            ``timestamp_utc``.

        Returns
        -------
        ``(n_marked, n_skipped)`` — counts of inserted vs. TTL-skipped rows.
        """
        n_marked = n_skipped = 0
        for rec in records:
            inserted = self.mark(
                subject_key   = str(rec.get("subject_key", "")),
                similarity    = float(rec.get("similarity", 0.0)),
                matched       = bool(rec.get("matched", False)),
                group_id      = str(rec.get("group_id", "")),
                camera_id     = str(rec.get("camera_id", "default")),
                timestamp_utc = rec.get("timestamp_utc") or None,
            )
            if inserted:
                n_marked += 1
            else:
                n_skipped += 1
        return n_marked, n_skipped

    # ──────────────────────────────────────────────────────────────────────────
    # Read / aggregate
    # ──────────────────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Total number of rows in the attendance table."""
        cur = self._conn.execute("SELECT COUNT(*) FROM attendance")
        return int(cur.fetchone()[0])

    def query_aggregated(
        self,
        *,
        from_ts: str | None = None,
        to_ts: str | None = None,
        group_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Return aggregated attendance statistics with optional filters.

        All parameters are optional.  When omitted, the corresponding dimension
        is not filtered.

        Parameters
        ----------
        from_ts:
            ISO-8601 lower bound (inclusive), e.g. ``"2026-02-28T00:00:00Z"``.
        to_ts:
            ISO-8601 upper bound (inclusive), e.g. ``"2026-02-28T23:59:59Z"``.
        group_id:
            Exact ``group_id`` to filter on.

        Returns
        -------
        ::

            {
              "total_marks":   int,
              "matched_marks": int,
              "session_ttl_s": int,
              "filters":       {"from": str|None, "to": str|None, "group_id": str|None},
              "per_subject":   [
                {"subject_key": str, "total": int, "matched": int, "last_seen": str},
                ...
              ],
              "per_camera":    [
                {"camera_id": str, "total": int, "matched": int,
                 "unique_subjects": int},
                ...
              ],
            }
        """
        params: list[Any] = []
        clauses: list[str] = []

        if from_ts:
            clauses.append("timestamp_utc >= ?")
            params.append(from_ts)
        if to_ts:
            clauses.append("timestamp_utc <= ?")
            params.append(to_ts)
        if group_id:
            clauses.append("group_id = ?")
            params.append(group_id)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        # ── Totals ─────────────────────────────────────────────────────────────
        cur = self._conn.execute(
            f"SELECT COUNT(*), COALESCE(SUM(matched), 0) FROM attendance {where}",
            params,
        )
        total_marks, matched_marks = cur.fetchone()

        # ── Per-subject aggregation ────────────────────────────────────────────
        cur = self._conn.execute(
            f"""
            SELECT subject_key,
                   COUNT(*)                    AS total,
                   COALESCE(SUM(matched), 0)   AS matched,
                   MAX(timestamp_utc)           AS last_seen
            FROM   attendance
            {where}
            GROUP BY subject_key
            ORDER BY last_seen DESC
            """,
            params,
        )
        per_subject = [
            {
                "subject_key": r["subject_key"],
                "total":       int(r["total"]),
                "matched":     int(r["matched"]),
                "last_seen":   r["last_seen"],
            }
            for r in cur.fetchall()
        ]

        # ── Per-camera aggregation ─────────────────────────────────────────────
        cur = self._conn.execute(
            f"""
            SELECT camera_id,
                   COUNT(*)                    AS total,
                   COALESCE(SUM(matched), 0)   AS matched,
                   COUNT(DISTINCT subject_key)  AS unique_subjects
            FROM   attendance
            {where}
            GROUP BY camera_id
            ORDER BY total DESC
            """,
            params,
        )
        per_camera = [
            {
                "camera_id":       r["camera_id"],
                "total":           int(r["total"]),
                "matched":         int(r["matched"]),
                "unique_subjects": int(r["unique_subjects"]),
            }
            for r in cur.fetchall()
        ]

        return {
            "total_marks":   int(total_marks),
            "matched_marks": int(matched_marks),
            "session_ttl_s": self._ttl,
            "filters": {
                "from":     from_ts,
                "to":       to_ts,
                "group_id": group_id,
            },
            "per_subject": per_subject,
            "per_camera":  per_camera,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Release the SQLite connection."""
        self._conn.close()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"AttendanceStore(path={self._path!r}, ttl={self._ttl}s,"
            f" rows={self.count()})"
        )
