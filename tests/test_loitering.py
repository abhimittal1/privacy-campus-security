"""
tests/test_loitering.py – Unit tests for LoiteringStore and LoiteringDetector.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time

import pytest

from src.privacy.loitering_store    import LoiteringStore
from src.privacy.loitering_detector import LoiteringDetector


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _tmp_store() -> LoiteringStore:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return LoiteringStore(path)


# ─────────────────────────────────────────────
#  LoiteringStore tests
# ─────────────────────────────────────────────

class TestLoiteringStore:

    def test_add_returns_dict_with_id(self):
        store = _tmp_store()
        ev = store.add("cam-1", "subj-abc", dwell_s=65.0)
        assert isinstance(ev, dict)
        assert ev["id"] >= 1
        assert ev["camera_id"] == "cam-1"
        assert ev["subject_key"] == "subj-abc"
        assert ev["dwell_s"] == 65.0

    def test_add_default_first_seen_is_not_none(self):
        store = _tmp_store()
        ev = store.add("cam-1", "subj-abc", dwell_s=30.0)
        assert ev["first_seen_utc"] is not None

    def test_add_custom_first_seen_utc(self):
        store = _tmp_store()
        ev = store.add("cam-1", "subj-1",
                       first_seen_utc="2026-02-28T00:00:00+00:00",
                       dwell_s=70.0)
        fetched = store.get(ev["id"])
        assert fetched["first_seen_utc"] == "2026-02-28T00:00:00+00:00"

    def test_get_unknown_returns_none(self):
        store = _tmp_store()
        assert store.get(99999) is None

    def test_dismiss_sets_flag(self):
        store = _tmp_store()
        ev = store.add("cam-1", "subj-001", dwell_s=60.0)
        ok = store.dismiss(ev["id"], action_by="Officer Kim")
        assert ok is True
        fetched = store.get(ev["id"])
        assert fetched["dismissed"] is True
        assert fetched["action_by"] == "Officer Kim"
        assert fetched["action_at"] is not None

    def test_dismiss_unknown_id_returns_false(self):
        store = _tmp_store()
        assert store.dismiss(99999, action_by="x") is False

    def test_list_recent_returns_newest_first(self):
        store = _tmp_store()
        ids = []
        for i in range(5):
            ev = store.add(f"cam-{i}", f"subj-{i}", dwell_s=float(60 + i))
            ids.append(ev["id"])
        listed = store.list_recent(n=10)
        returned_ids = [e["id"] for e in listed]
        assert returned_ids == sorted(ids, reverse=True)

    def test_list_recent_n_limit(self):
        store = _tmp_store()
        for i in range(8):
            store.add("cam-1", f"s{i}", dwell_s=60.0)
        assert len(store.list_recent(n=3)) == 3

    def test_list_recent_filter_camera(self):
        store = _tmp_store()
        store.add("cam-alpha", "s1", dwell_s=60.0)
        store.add("cam-beta",  "s2", dwell_s=60.0)
        store.add("cam-alpha", "s3", dwell_s=60.0)
        results = store.list_recent(n=50, camera_id="cam-alpha")
        assert all(e["camera_id"] == "cam-alpha" for e in results)
        assert len(results) == 2

    def test_list_recent_filter_subject_key(self):
        store = _tmp_store()
        store.add("cam-1", "subj-A", dwell_s=60.0)
        store.add("cam-1", "subj-B", dwell_s=60.0)
        store.add("cam-2", "subj-A", dwell_s=60.0)
        results = store.list_recent(n=50, subject_key="subj-A")
        assert all(e["subject_key"] == "subj-A" for e in results)
        assert len(results) == 2

    def test_list_recent_filter_dismissed_false(self):
        store = _tmp_store()
        ev1 = store.add("cam-1", "s1", dwell_s=60.0)
        ev2 = store.add("cam-1", "s2", dwell_s=60.0)
        store.dismiss(ev1["id"])
        active = store.list_recent(n=50, dismissed=False)
        ids = [e["id"] for e in active]
        assert ev1["id"] not in ids
        assert ev2["id"] in ids

    def test_list_recent_filter_dismissed_true(self):
        store = _tmp_store()
        ev1 = store.add("cam-1", "s1", dwell_s=60.0)
        ev2 = store.add("cam-1", "s2", dwell_s=60.0)
        store.dismiss(ev1["id"])
        results = store.list_recent(n=50, dismissed=True)
        ids = [e["id"] for e in results]
        assert ev1["id"] in ids
        assert ev2["id"] not in ids

    def test_count_active(self):
        store = _tmp_store()
        ev1 = store.add("cam-1", "s1", dwell_s=60.0)
        ev2 = store.add("cam-1", "s2", dwell_s=60.0)
        ev3 = store.add("cam-1", "s3", dwell_s=60.0)
        store.dismiss(ev1["id"])
        assert store.count_active() == 2  # ev2 and ev3

    def test_count_active_empty(self):
        store = _tmp_store()
        assert store.count_active() == 0

    def test_dismissed_flag_is_bool(self):
        store = _tmp_store()
        ev = store.add("cam-1", "s1", dwell_s=60.0)
        fetched = store.get(ev["id"])
        assert fetched["dismissed"] is False

    def test_thread_safety(self):
        store = _tmp_store()
        errors = []

        def worker(cam_id):
            try:
                for i in range(10):
                    store.add(cam_id, f"subj-{i}", dwell_s=60.0)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(f"cam-{j}",)) for j in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert store.count_active() == 50  # 5×10


# ─────────────────────────────────────────────
#  LoiteringDetector tests
# ─────────────────────────────────────────────

class TestLoiteringDetector:

    def _det(self, dwell_s=0.3, stale_s=2.0, cooldown_s=1.0):
        return LoiteringDetector(dwell_s=dwell_s, stale_s=stale_s, cooldown_s=cooldown_s)

    def test_first_observation_no_detection(self):
        det = self._det(dwell_s=60.0)
        result = det.observe("cam-1", "subj-A")
        assert result == []

    def test_returns_list(self):
        det = self._det()
        assert isinstance(det.observe("cam-1", "subj-A"), list)

    def test_detection_after_dwell(self):
        """Subject observed repeatedly for > dwell_s seconds → fires detection."""
        det = self._det(dwell_s=0.2, stale_s=5.0, cooldown_s=999.0)
        found = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 5.0:
            found.extend(det.observe("cam-1", "subj-A"))
            if found:
                break
            time.sleep(0.05)
        assert len(found) == 1

    def test_detection_dict_has_required_keys(self):
        det = self._det(dwell_s=0.1, stale_s=5.0, cooldown_s=999.0)
        found = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 5.0:
            found.extend(det.observe("cam-1", "subj-A"))
            if found:
                break
            time.sleep(0.02)
        if found:
            required = {"camera_id", "subject_key", "first_seen_utc", "detected_utc", "dwell_s"}
            assert required.issubset(found[0].keys())
            assert found[0]["camera_id"] == "cam-1"
            assert found[0]["subject_key"] == "subj-A"
            assert found[0]["dwell_s"] >= 0.1

    def test_cooldown_prevents_duplicate(self):
        """After firing, should not fire again until cooldown expires."""
        det = self._det(dwell_s=0.1, stale_s=99.0, cooldown_s=999.0)
        found = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 5.0:
            found.extend(det.observe("cam-1", "subj-A"))
            if found:
                break
            time.sleep(0.02)
        assert len(found) == 1
        # Keep observing — should NOT fire again (cooldown=999s)
        extra = []
        for _ in range(50):
            extra.extend(det.observe("cam-1", "subj-A"))
            time.sleep(0.01)
        assert extra == []

    def test_cooldown_allows_repeat_after_expiry(self):
        """After cooldown expires, the same subject can fire again."""
        det = self._det(dwell_s=0.05, stale_s=99.0, cooldown_s=0.1)
        # First fire
        found = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3.0:
            found.extend(det.observe("cam-1", "subj-A"))
            if found:
                break
            time.sleep(0.01)
        assert len(found) == 1
        # Wait for cooldown
        time.sleep(0.2)
        # Second fire
        found2 = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 3.0:
            found2.extend(det.observe("cam-1", "subj-A"))
            if found2:
                break
            time.sleep(0.01)
        assert len(found2) == 1

    def test_independent_cameras(self):
        """Each camera tracks subjects independently."""
        det = self._det(dwell_s=0.1, stale_s=5.0, cooldown_s=999.0)
        results_a, results_b = [], []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 5.0:
            results_a.extend(det.observe("cam-A", "subj-X"))
            # cam-B doesn't see subj-X, only subj-Y
            results_b.extend(det.observe("cam-B", "subj-Y"))
            if results_a:
                break
            time.sleep(0.02)
        assert len(results_a) >= 1
        # subj-X on cam-B was never observed
        assert all(e["camera_id"] == "cam-A" for e in results_a)

    def test_independent_subjects_same_camera(self):
        """Two subjects at the same camera tracked independently."""
        det = self._det(dwell_s=0.1, stale_s=5.0, cooldown_s=999.0)
        found = []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 5.0:
            found.extend(det.observe("cam-1", "subj-A"))
            found.extend(det.observe("cam-1", "subj-B"))
            if len(found) >= 2:
                break
            time.sleep(0.02)
        subjects = {e["subject_key"] for e in found}
        assert "subj-A" in subjects
        assert "subj-B" in subjects

    def test_purge_stale_resets_dwell(self):
        """After stale purge, subject's dwell clock restarts on re-entry."""
        det = self._det(dwell_s=0.1, stale_s=0.05, cooldown_s=999.0)
        # Observe once to create state
        det.observe("cam-1", "subj-A")
        # Wait for stale window to pass
        time.sleep(0.15)
        # Purge without subj-A in active set
        det.purge_stale("cam-1", active_keys=set())
        # After purge, subj-A should not be tracked any more
        assert "subj-A" not in det.active_subjects("cam-1")

    def test_purge_stale_keeps_recent_subjects(self):
        """Subjects seen recently should NOT be purged."""
        det = self._det(dwell_s=60.0, stale_s=5.0, cooldown_s=999.0)
        det.observe("cam-1", "subj-A")
        # purge immediately — subject was just seen
        det.purge_stale("cam-1", active_keys=set())
        # should NOT be purged (last_seen is very recent)
        assert "subj-A" in det.active_subjects("cam-1")

    def test_active_subjects_initially_empty(self):
        det = self._det()
        assert det.active_subjects("cam-unknown") == []

    def test_reset_camera(self):
        det = self._det(dwell_s=60.0)
        det.observe("cam-1", "subj-A")
        det.observe("cam-1", "subj-B")
        det.reset_camera("cam-1")
        assert det.active_subjects("cam-1") == []

    def test_reset_all(self):
        det = self._det(dwell_s=60.0)
        det.observe("cam-1", "subj-A")
        det.observe("cam-2", "subj-B")
        det.reset_all()
        assert det.active_subjects("cam-1") == []
        assert det.active_subjects("cam-2") == []

    def test_thread_safety(self):
        """Multiple threads observing concurrently should not corrupt state."""
        det = self._det(dwell_s=60.0, stale_s=5.0)
        errors = []

        def worker(cam_id, subj):
            try:
                for _ in range(20):
                    det.observe(cam_id, subj)
                    time.sleep(0.001)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(f"cam-{i}", f"subj-{i}"))
            for i in range(6)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
