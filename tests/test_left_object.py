"""
tests/test_left_object.py – Unit tests for LeftObjectStore and LeftObjectDetector.
"""

from __future__ import annotations

import struct
import zlib
import time
import tempfile
import os
from pathlib import Path

import pytest

from src.privacy.left_object_store import LeftObjectStore


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _tmp_store() -> LeftObjectStore:
    """Return a LeftObjectStore backed by an in-memory (temp) SQLite database."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = LeftObjectStore(path)
    return store


def _minimal_png(w: int = 8, h: int = 8) -> bytes:
    """Create a minimal valid 8-bit greyscale PNG in memory."""
    def _chunk(name: bytes, data: bytes) -> bytes:
        c = zlib.crc32(name + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", c)

    raw_rows = b"".join(
        b"\x00" + bytes([0] * w) for _ in range(h)
    )
    compressed = zlib.compress(raw_rows)
    png  = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
    png += _chunk(b"IDAT", compressed)
    png += _chunk(b"IEND", b"")
    return png


# ─────────────────────────────────────────────
#  LeftObjectStore tests
# ─────────────────────────────────────────────

class TestLeftObjectStore:

    def test_add_returns_dict_with_id(self):
        store = _tmp_store()
        ev = store.add("cam-1", bbox=[10, 20, 30, 40], dwell_s=32.0)
        assert isinstance(ev, dict)
        assert "id" in ev
        assert ev["id"] >= 1
        assert ev["camera_id"] == "cam-1"

    def test_add_bbox_round_trips(self):
        store = _tmp_store()
        bbox_in = [5, 10, 80, 60]
        ev = store.add("cam-1", bbox=bbox_in, dwell_s=31.0)
        fetched = store.get(ev["id"])
        assert fetched["bbox"] == bbox_in

    def test_add_without_snapshot(self):
        store = _tmp_store()
        ev = store.add("cam-2", bbox=[0, 0, 100, 100], dwell_s=35.0)
        assert ev["snapshot_b64"] is None

    def test_add_with_snapshot(self):
        store = _tmp_store()
        fake_uri = "data:image/jpeg;base64,/9j/abc"
        ev = store.add("cam-2", bbox=[0, 0, 10, 10], dwell_s=40.0, snapshot_b64=fake_uri)
        fetched = store.get(ev["id"])
        assert fetched["snapshot_b64"] == fake_uri

    def test_get_unknown_returns_none(self):
        store = _tmp_store()
        assert store.get(99999) is None

    def test_dismiss_sets_flag(self):
        store = _tmp_store()
        ev = store.add("cam-1", bbox=[0, 0, 1, 1], dwell_s=30.0)
        result = store.dismiss(ev["id"], action_by="Officer A")
        assert result is True
        fetched = store.get(ev["id"])
        assert fetched["dismissed"] == 1
        assert fetched["action_by"] == "Officer A"
        assert fetched["action_at"] is not None

    def test_confirm_sets_flag(self):
        store = _tmp_store()
        ev = store.add("cam-1", bbox=[0, 0, 1, 1], dwell_s=30.0)
        result = store.confirm(ev["id"], action_by="Admin B")
        assert result is True
        fetched = store.get(ev["id"])
        assert fetched["confirmed"] == 1
        assert fetched["action_by"] == "Admin B"

    def test_dismiss_unknown_id_returns_false(self):
        store = _tmp_store()
        assert store.dismiss(99999, action_by="x") is False

    def test_confirm_unknown_id_returns_false(self):
        store = _tmp_store()
        assert store.confirm(99999, action_by="x") is False

    def test_list_recent_returns_newest_first(self):
        store = _tmp_store()
        ids = []
        for i in range(5):
            ev = store.add(f"cam-{i}", bbox=[i, i, 10, 10], dwell_s=float(i + 30))
            ids.append(ev["id"])
        listed = store.list_recent(n=10)
        returned_ids = [e["id"] for e in listed]
        assert returned_ids == sorted(ids, reverse=True)

    def test_list_recent_n_limit(self):
        store = _tmp_store()
        for i in range(8):
            store.add("cam-1", bbox=[0, 0, 1, 1], dwell_s=30.0)
        assert len(store.list_recent(n=3)) == 3

    def test_list_recent_filter_camera(self):
        store = _tmp_store()
        store.add("cam-alpha", bbox=[0, 0, 1, 1], dwell_s=30.0)
        store.add("cam-beta",  bbox=[0, 0, 1, 1], dwell_s=30.0)
        store.add("cam-alpha", bbox=[0, 0, 1, 1], dwell_s=30.0)
        results = store.list_recent(n=50, camera_id="cam-alpha")
        assert all(e["camera_id"] == "cam-alpha" for e in results)
        assert len(results) == 2

    def test_list_recent_filter_dismissed_false(self):
        store = _tmp_store()
        ev1 = store.add("cam-1", bbox=[0, 0, 1, 1], dwell_s=30.0)
        ev2 = store.add("cam-1", bbox=[0, 0, 1, 1], dwell_s=31.0)
        store.dismiss(ev1["id"], action_by="op")
        active = store.list_recent(n=50, dismissed=False)
        ids = [e["id"] for e in active]
        assert ev1["id"] not in ids
        assert ev2["id"] in ids

    def test_list_recent_filter_confirmed_true(self):
        store = _tmp_store()
        ev1 = store.add("cam-1", bbox=[0, 0, 1, 1], dwell_s=30.0)
        ev2 = store.add("cam-1", bbox=[0, 0, 1, 1], dwell_s=31.0)
        store.confirm(ev1["id"], action_by="op")
        confirmed = store.list_recent(n=50, confirmed=True)
        ids = [e["id"] for e in confirmed]
        assert ev1["id"] in ids
        assert ev2["id"] not in ids

    def test_count_active_only_counts_unactioned(self):
        store = _tmp_store()
        ev1 = store.add("cam-1", bbox=[0, 0, 1, 1], dwell_s=30.0)
        ev2 = store.add("cam-1", bbox=[0, 0, 1, 1], dwell_s=31.0)
        ev3 = store.add("cam-1", bbox=[0, 0, 1, 1], dwell_s=32.0)
        store.dismiss(ev1["id"])
        store.confirm(ev2["id"])
        assert store.count_active() == 1  # only ev3 remains active

    def test_count_active_empty(self):
        store = _tmp_store()
        assert store.count_active() == 0

    def test_multiple_cameras_independent(self):
        store = _tmp_store()
        for cam in ["north", "south", "east"]:
            for _ in range(3):
                store.add(cam, bbox=[0, 0, 10, 10], dwell_s=30.0)
        for cam in ["north", "south", "east"]:
            results = store.list_recent(n=50, camera_id=cam)
            assert len(results) == 3

    def test_thread_safety(self):
        """Multiple threads should be able to add events concurrently without errors."""
        import threading
        store = _tmp_store()
        errors = []

        def worker(cam_id):
            try:
                for i in range(10):
                    store.add(cam_id, bbox=[i, i, 10, 10], dwell_s=30.0)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(f"cam-{j}",)) for j in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        assert store.count_active() == 50  # 5 threads × 10 events each


# ─────────────────────────────────────────────
#  LeftObjectDetector tests
# ─────────────────────────────────────────────

class TestLeftObjectDetector:
    """Tests for LeftObjectDetector that run without cv2."""

    def _make_detector(self, dwell_s: float = 0.5, stale_s: float = 2.0,
                       cooldown_s: float = 1.0, alpha_slow: float = 0.002,
                       alpha_fast: float = 0.05):
        from src.privacy.left_object_detector import LeftObjectDetector
        return LeftObjectDetector(
            dwell_s=dwell_s,
            fg_thresh=22.0,
            min_blob_frac=0.001,
            stale_s=stale_s,
            cooldown_s=cooldown_s,
            alpha_slow=alpha_slow,
            alpha_fast=alpha_fast,
            proc_w=64,
            proc_h=48,
        )

    def _blank_png(self, w: int = 64, h: int = 48, value: int = 100) -> bytes:
        """Return a solid-colour greyscale PNG."""
        import struct, zlib
        raw = b"".join(b"\x00" + bytes([value] * w) for _ in range(h))
        compressed = zlib.compress(raw)

        def chunk(name: bytes, data: bytes) -> bytes:
            crc = zlib.crc32(name + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + name + data + struct.pack(">I", crc)

        png  = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
        png += chunk(b"IDAT", compressed)
        png += chunk(b"IEND", b"")
        return png

    def _bright_rect_png(self, w: int = 64, h: int = 48,
                         bg: int = 100, obj: int = 210,
                         rx: int = 20, ry: int = 15, rw: int = 24, rh: int = 18) -> bytes:
        """A PNG with a bright rectangle on a flat background."""
        import struct, zlib
        pixels = [[bg] * w for _ in range(h)]
        for row in range(ry, ry + rh):
            for col in range(rx, rx + rw):
                pixels[row][col] = obj
        raw = b"".join(b"\x00" + bytes(row) for row in pixels)
        compressed = zlib.compress(raw)

        def chunk(name: bytes, data: bytes) -> bytes:
            crc = zlib.crc32(name + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + name + data + struct.pack(">I", crc)

        png  = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
        png += chunk(b"IDAT", compressed)
        png += chunk(b"IEND", b"")
        return png

    def test_blank_frames_no_detection(self):
        det = self._make_detector(dwell_s=0.1)
        blank = self._blank_png()
        detections = []
        for _ in range(20):
            detections.extend(det.process_frame("cam-1", blank))
        assert detections == [], "Uniform frames should not trigger detections"

    def test_process_frame_returns_list(self):
        det = self._make_detector()
        result = det.process_frame("cam-1", self._blank_png())
        assert isinstance(result, list)

    def test_static_object_detected_after_dwell(self):
        """Feed background frames to stabilise bg, then add a bright rectangle.
        After enough frames the detector should fire a left_object detection."""
        det = self._make_detector(dwell_s=0.3, cooldown_s=0.0, alpha_slow=0.01, alpha_fast=0.15)
        blank = self._blank_png()
        rect  = self._bright_rect_png()

        # Warm up background model with ~60 blank frames
        for _ in range(60):
            det.process_frame("warm", blank)

        # Now feed bright-rectangle frames long enough to exceed dwell_s
        found = []
        t0 = time.monotonic()
        # Feed for up to 5 seconds at simulated speed (each frame ~0.05s real time)
        for _ in range(200):
            results = det.process_frame("warm", rect)
            found.extend(results)
            if found:
                break
            time.sleep(0.005)  # keep test fast

        assert len(found) >= 1, "Expected at least one left-object detection after dwell period"
        det0 = found[0]
        assert det0["camera_id"] == "warm"
        assert "bbox" in det0
        assert "dwell_s" in det0
        assert "detected_utc" in det0
        assert det0["dwell_s"] >= 0

    def test_detection_dict_has_required_keys(self):
        det = self._make_detector(dwell_s=0.1, cooldown_s=0.0, alpha_slow=0.05, alpha_fast=0.3)
        blank = self._blank_png()
        rect  = self._bright_rect_png()
        for _ in range(30):
            det.process_frame("cam-x", blank)
        found = []
        for _ in range(100):
            r = det.process_frame("cam-x", rect)
            found.extend(r)
            if found:
                break
            time.sleep(0.005)
        if found:
            required = {"camera_id", "bbox", "first_seen_utc", "detected_utc", "dwell_s"}
            assert required.issubset(found[0].keys())

    def test_independent_camera_states(self):
        """Two cameras should maintain independent background models."""
        det = self._make_detector(dwell_s=0.1, alpha_slow=0.05, alpha_fast=0.3)
        blank = self._blank_png()
        rect  = self._bright_rect_png()
        # Warm both cameras
        for _ in range(30):
            det.process_frame("cam-A", blank)
            det.process_frame("cam-B", blank)
        # Feed rect to cam-A only
        found_A, found_B = [], []
        for _ in range(100):
            found_A.extend(det.process_frame("cam-A", rect))
            found_B.extend(det.process_frame("cam-B", blank))
            if found_A:
                break
            time.sleep(0.005)
        # cam-A should detect; cam-B (still receiving blank) should not
        assert len(found_A) >= 1
        # cam-B blank frames: no detections during the same window
        assert len(found_B) == 0

    def test_cooldown_prevents_duplicate_events(self):
        """After firing, the detector should not fire again immediately."""
        det = self._make_detector(dwell_s=0.1, cooldown_s=999.0,
                                  alpha_slow=0.05, alpha_fast=0.3)
        blank = self._blank_png()
        rect  = self._bright_rect_png()
        for _ in range(30):
            det.process_frame("cam-c", blank)
        found = []
        for _ in range(200):
            found.extend(det.process_frame("cam-c", rect))
            time.sleep(0.005)
        # With a large cooldown, only one detection should be emitted
        assert len(found) == 1, f"Expected 1 detection (cooldown=999s) but got {len(found)}"

    def test_invalid_frame_bytes_returns_empty(self):
        """Non-image bytes should not crash the detector; it should return []."""
        det = self._make_detector()
        result = det.process_frame("cam-bad", b"not an image at all")
        assert result == []

    def test_empty_frame_bytes_returns_empty(self):
        det = self._make_detector()
        result = det.process_frame("cam-bad", b"")
        assert result == []
