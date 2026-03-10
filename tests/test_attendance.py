"""
test_attendance.py – Unit tests for AttendanceStore.

Covers:
  * Table creation and basic mark/count round-trip.
  * Session TTL deduplication (duplicate within TTL → skipped).
  * TTL expiry (mark after TTL elapsed → inserted).
  * mark_many: returns (n_marked, n_skipped).
  * query_aggregated: total counts, per_subject, per_camera.
  * query_aggregated: time-range filter.
  * query_aggregated: group_id filter.
  * mark() raises ValueError for empty subject_key.
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.privacy.attendance_store import AttendanceStore


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _ts(offset_seconds: int = 0) -> str:
    """Return an ISO-8601 UTC timestamp shifted by *offset_seconds* from now."""
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).isoformat()


@pytest.fixture
def att(tmp_path: Path) -> AttendanceStore:
    """A fresh AttendanceStore backed by a temp SQLite file (TTL = 300 s)."""
    return AttendanceStore(tmp_path / "attendance.db", session_ttl_s=300)


@pytest.fixture
def att_short_ttl(tmp_path: Path) -> AttendanceStore:
    """A fresh AttendanceStore with a very short TTL (1 s) for expiry tests."""
    return AttendanceStore(tmp_path / "att_short.db", session_ttl_s=1)


# ─── Basic functionality ───────────────────────────────────────────────────────


class TestBasicMark:
    def test_count_starts_at_zero(self, att: AttendanceStore) -> None:
        assert att.count() == 0

    def test_single_mark_inserts_row(self, att: AttendanceStore) -> None:
        inserted = att.mark(subject_key="subj-1", similarity=0.85, matched=True)
        assert inserted is True
        assert att.count() == 1

    def test_mark_returns_false_within_ttl(self, att: AttendanceStore) -> None:
        att.mark(subject_key="subj-1", similarity=0.85, matched=True)
        # Second mark within TTL for same (subject_key, camera_id) → skipped.
        inserted = att.mark(subject_key="subj-1", similarity=0.90, matched=True)
        assert inserted is False
        assert att.count() == 1

    def test_mark_different_camera_not_deduped(self, att: AttendanceStore) -> None:
        att.mark(subject_key="subj-1", similarity=0.85, matched=True,
                 camera_id="cam-1")
        # Different camera → independent TTL bucket → should insert.
        inserted = att.mark(subject_key="subj-1", similarity=0.85, matched=True,
                            camera_id="cam-2")
        assert inserted is True
        assert att.count() == 2

    def test_mark_empty_subject_key_raises(self, att: AttendanceStore) -> None:
        with pytest.raises(ValueError):
            att.mark(subject_key="", similarity=0.5, matched=False)


class TestTtlExpiry:
    def test_mark_after_ttl_inserts_again(self, att_short_ttl: AttendanceStore) -> None:
        att = att_short_ttl
        # Mark with a timestamp 2 s in the past (older than the 1-s TTL).
        past_ts = _ts(-2)
        att.mark(subject_key="subj-x", similarity=0.7, matched=False,
                 timestamp_utc=past_ts)
        # Now mark again at current time → TTL expired → should insert.
        inserted = att.mark(subject_key="subj-x", similarity=0.8, matched=True)
        assert inserted is True
        assert att.count() == 2


# ─── mark_many ────────────────────────────────────────────────────────────────


class TestMarkMany:
    def test_mark_many_all_fresh(self, att: AttendanceStore) -> None:
        records = [
            {"subject_key": f"s{i}", "similarity": 0.75 + i * 0.01, "matched": True}
            for i in range(6)
        ]
        n_marked, n_skipped = att.mark_many(records)
        assert n_marked == 6
        assert n_skipped == 0
        assert att.count() == 6

    def test_mark_many_with_duplicates(self, att: AttendanceStore) -> None:
        records = [
            {"subject_key": "dup", "similarity": 0.8, "matched": True},
            {"subject_key": "dup", "similarity": 0.9, "matched": True},  # dup
            {"subject_key": "unique", "similarity": 0.6, "matched": False},
        ]
        n_marked, n_skipped = att.mark_many(records)
        assert n_marked == 2
        assert n_skipped == 1
        assert att.count() == 2


# ─── query_aggregated ─────────────────────────────────────────────────────────


class TestQueryAggregated:
    @pytest.fixture
    def populated(self, tmp_path: Path) -> AttendanceStore:
        """AttendanceStore pre-populated with 6 marks across 2 groups & 2 cams."""
        att = AttendanceStore(tmp_path / "pop.db", session_ttl_s=0)
        # TTL = 0 disables dedup — every mark inserts regardless.
        records = [
            # group SHIFT-A, cam-1: 3 subjects, all matched
            {"subject_key": "a", "similarity": 0.9, "matched": True,
             "group_id": "SHIFT-A", "camera_id": "cam-1",
             "timestamp_utc": "2026-02-28T08:00:00+00:00"},
            {"subject_key": "b", "similarity": 0.8, "matched": True,
             "group_id": "SHIFT-A", "camera_id": "cam-1",
             "timestamp_utc": "2026-02-28T08:01:00+00:00"},
            {"subject_key": "c", "similarity": 0.7, "matched": True,
             "group_id": "SHIFT-A", "camera_id": "cam-1",
             "timestamp_utc": "2026-02-28T08:02:00+00:00"},
            # group SHIFT-B, cam-2: 2 subjects, 1 matched
            {"subject_key": "d", "similarity": 0.5, "matched": False,
             "group_id": "SHIFT-B", "camera_id": "cam-2",
             "timestamp_utc": "2026-02-28T14:00:00+00:00"},
            {"subject_key": "e", "similarity": 0.85, "matched": True,
             "group_id": "SHIFT-B", "camera_id": "cam-2",
             "timestamp_utc": "2026-02-28T14:01:00+00:00"},
            # group SHIFT-A again, cam-2 (cross-camera mark for "a")
            {"subject_key": "a", "similarity": 0.88, "matched": True,
             "group_id": "SHIFT-A", "camera_id": "cam-2",
             "timestamp_utc": "2026-02-28T09:00:00+00:00"},
        ]
        att.mark_many(records)
        return att

    def test_total_counts(self, populated: AttendanceStore) -> None:
        r = populated.query_aggregated()
        assert r["total_marks"] == 6
        assert r["matched_marks"] == 5

    def test_per_subject_length(self, populated: AttendanceStore) -> None:
        r = populated.query_aggregated()
        # 5 unique subjects: a, b, c, d, e
        assert len(r["per_subject"]) == 5

    def test_subject_a_has_two_marks(self, populated: AttendanceStore) -> None:
        r = populated.query_aggregated()
        subj_a = next(s for s in r["per_subject"] if s["subject_key"] == "a")
        assert subj_a["total"] == 2
        assert subj_a["matched"] == 2

    def test_per_camera_counts(self, populated: AttendanceStore) -> None:
        r = populated.query_aggregated()
        by_cam = {c["camera_id"]: c for c in r["per_camera"]}
        assert by_cam["cam-1"]["total"] == 3
        assert by_cam["cam-2"]["total"] == 3
        assert by_cam["cam-1"]["unique_subjects"] == 3
        assert by_cam["cam-2"]["unique_subjects"] == 3   # a, d, e

    def test_group_id_filter(self, populated: AttendanceStore) -> None:
        r = populated.query_aggregated(group_id="SHIFT-A")
        assert r["total_marks"] == 4   # a(cam-1), b, c, a(cam-2)
        assert r["matched_marks"] == 4

    def test_from_ts_filter(self, populated: AttendanceStore) -> None:
        # Only afternoon marks (>= 14:00)
        r = populated.query_aggregated(from_ts="2026-02-28T14:00:00+00:00")
        assert r["total_marks"] == 2   # d, e

    def test_to_ts_filter(self, populated: AttendanceStore) -> None:
        r = populated.query_aggregated(to_ts="2026-02-28T09:00:00+00:00")
        assert r["total_marks"] == 4   # a(08:00), b(08:01), c(08:02), a(09:00)

    def test_filters_echoed_in_response(self, populated: AttendanceStore) -> None:
        r = populated.query_aggregated(from_ts="2026-02-28T00:00:00Z",
                                       group_id="SHIFT-B")
        assert r["filters"]["from"] == "2026-02-28T00:00:00Z"
        assert r["filters"]["group_id"] == "SHIFT-B"
        assert r["filters"]["to"] is None

    def test_empty_store_returns_zeros(self, att: AttendanceStore) -> None:
        r = att.query_aggregated()
        assert r["total_marks"] == 0
        assert r["per_subject"] == []
        assert r["per_camera"] == []

    def test_session_ttl_s_present(self, att: AttendanceStore) -> None:
        r = att.query_aggregated()
        assert "session_ttl_s" in r
        assert r["session_ttl_s"] == 300


# ─── Persistence (same-file coexistence with embeddings table) ────────────────


class TestPersistence:
    def test_table_coexists_with_embeddings_schema(self, tmp_path: Path) -> None:
        """AttendanceStore must work even when the embeddings table already exists."""
        db_path = tmp_path / "shared.db"
        # Create the embeddings table as EmbeddingStore does.
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
                hashed_id TEXT PRIMARY KEY, embedding BLOB NOT NULL, dim INTEGER NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

        att = AttendanceStore(db_path)
        inserted = att.mark(subject_key="shared-test", similarity=0.9, matched=True)
        assert inserted is True
        assert att.count() == 1
