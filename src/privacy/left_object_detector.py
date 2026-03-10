"""
left_object_detector.py – Background-subtraction pipeline for unattended-object detection.

Algorithm
---------
Each camera maintains two exponential-moving-average (EMA) backgrounds in grayscale:

* **slow_bg** – integrates only *background* pixels (α ≈ 0.002, ~500-frame memory).
  New static objects are visible in the diff against ``slow_bg``.
* **fast_bg** – integrates every pixel (α ≈ 0.05, ~20-frame memory).
  Pixels that are still changing appear in the diff against ``fast_bg``.

Left-object mask = ``(|frame − slow_bg| > fg_thresh)  AND  NOT  (|frame − fast_bg| > fg_thresh)``

Translation: something is present w.r.t. the long-term background *and* is no longer
moving → it was put down (left behind).

Candidate tracking
------------------
Foreground regions are connected-component-labelled (or OpenCV-contoured when available).
Each candidate (blob) is tracked by IoU overlap across frames:

* Candidate appears → ``first_seen_s`` stamped.
* Candidate persists for ``dwell_s`` seconds → **left-object event fired once**.
* Candidate disappears for ``stale_s`` seconds → removed from candidates.
* A per-camera cooldown prevents duplicate events for the same lingering object.

The detector is stateless from the caller's perspective; all state lives in
``_CameraState`` instances held in ``LeftObjectDetector._states``.

Dependencies
------------
Hard: numpy, Pillow (already in requirements.txt).
Soft: opencv-python (``cv2``).  If absent, morphological cleanup is skipped but the
numeric dual-background approach still works.

Usage
-----
::

    from src.privacy.left_object_detector import LeftObjectDetector

    detector = LeftObjectDetector(dwell_s=30.0)

    # called once per frame (from MQTT subscriber thread):
    detections = detector.process_frame(camera_id, frame_bytes)
    for det in detections:
        # det = {camera_id, bbox, first_seen_utc, detected_utc, dwell_s, snapshot_b64}
        store.add(det["camera_id"], bbox=det["bbox"], ...)
"""

from __future__ import annotations

import base64
import io
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

try:
    from PIL import Image as _PIL_Image  # type: ignore[import]
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

try:
    import cv2  # type: ignore[import]
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

log = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class _BlobCandidate:
    """One tracked foreground blob within a camera."""
    blob_id: int
    bbox: list[int]          # [x, y, w, h] in *processed-frame* coordinates
    first_seen_s: float      # time.monotonic() when first detected
    last_seen_s: float       # last time this blob was visible in processed frame
    fired: bool = False      # True once we have emitted the dwell event
    cooldown_until_s: float = 0.0  # after fire, re-arm after cooldown


@dataclass
class _CameraState:
    bg_slow: Optional[np.ndarray] = None   # EMA background, slow (float32 H×W)
    bg_fast: Optional[np.ndarray] = None   # EMA background, fast (float32 H×W)
    candidates: dict[int, _BlobCandidate] = field(default_factory=dict)
    next_blob_id: int = 0
    proc_w: int = 320
    proc_h: int = 180
    lock: threading.Lock = field(default_factory=threading.Lock)


def _iou(a: list[int], b: list[int]) -> float:
    """Compute intersection-over-union of two [x,y,w,h] boxes."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1 = max(ax1, bx1); ix2 = min(ax2, bx2)
    iy1 = max(ay1, by1); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _decode_frame(frame_bytes: bytes, target_w: int, target_h: int
                  ) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Decode a JPEG byte-string and return ``(gray_small_f32, orig_bgr)``.

    ``gray_small_f32`` is resized to (target_h, target_w) and cast to float32.
    ``orig_bgr`` is the full-resolution numpy array (H,W,3) uint8.

    Returns ``None`` on decode failure or if PIL is not installed.
    """
    if not _HAS_PIL:
        return None
    try:
        pil = _PIL_Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        orig_np = np.array(pil, dtype=np.uint8)      # (H, W, 3) RGB
        small   = pil.resize((target_w, target_h), _PIL_Image.BILINEAR)
        small_g = np.array(small.convert("L"), dtype=np.float32)  # (H, W)
        return small_g, orig_np
    except Exception as exc:
        log.debug("[LOD] frame decode failed: %s", exc)
        return None


def _morpho_clean(mask: np.ndarray) -> np.ndarray:
    """Morphological open → dilate to clean up the binary foreground mask."""
    if _HAS_CV2:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask   = cv2.dilate(mask, kernel2, iterations=1)
    else:
        # Minimal erosion/dilation via max-pooling without scipy
        # (acceptable quality; still suppresses tiny noise blobs)
        from numpy.lib.stride_tricks import as_strided
        k = 5
        pad = k // 2
        m = np.pad(mask.astype(np.float32), pad, mode="constant")
        shape = (mask.shape[0], mask.shape[1], k, k)
        strides = (m.strides[0], m.strides[1], m.strides[0], m.strides[1])
        view = as_strided(m, shape=shape, strides=strides)
        mask = (view.min(axis=(2, 3)) > 0).astype(np.uint8)   # erosion
        m2 = np.pad(mask.astype(np.float32), pad, mode="constant")
        view2 = as_strided(m2, shape=shape, strides=(m2.strides[0], m2.strides[1],
                                                      m2.strides[0], m2.strides[1]))
        mask = (view2.max(axis=(2, 3)) > 0).astype(np.uint8)  # dilation
    return mask


def _find_blobs(mask: np.ndarray) -> list[list[int]]:
    """
    Return a list of ``[x, y, w, h]`` bounding boxes for connected components
    in the binary mask, using cv2 if available, else a fast numpy fallback.
    """
    if _HAS_CV2:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [list(cv2.boundingRect(c)) for c in contours]  # [x,y,w,h]

    # Fallback: label connected components via scipy if present, else row-scan
    try:
        from scipy.ndimage import label as _label  # type: ignore[import]
        labeled, n_comp = _label(mask)
        bboxes = []
        for i in range(1, n_comp + 1):
            positions = np.argwhere(labeled == i)
            if len(positions) == 0:
                continue
            y0, x0 = positions.min(axis=0)
            y1, x1 = positions.max(axis=0)
            bboxes.append([int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)])
        return bboxes
    except ImportError:
        pass

    # Last resort: sliced row/col projection to find one big bounding box
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return []
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return [[int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]]


def _crop_b64(orig_rgb: np.ndarray,
              proc_bbox: list[int],
              proc_w: int, proc_h: int) -> str:
    """
    Crop the bounding box region from the original-resolution RGB frame and
    return a ``data:image/jpeg;base64,…`` string.

    ``proc_bbox`` is in processed-frame coords; we scale up to original size.
    """
    if not _HAS_PIL:
        return ""
    orig_h, orig_w = orig_rgb.shape[:2]
    sx = orig_w / proc_w
    sy = orig_h / proc_h
    x, y, w, h = proc_bbox
    pad = max(10, int(min(w, h) * 0.15))
    x1 = max(0, int(x * sx) - pad)
    y1 = max(0, int(y * sy) - pad)
    x2 = min(orig_w, int((x + w) * sx) + pad)
    y2 = min(orig_h, int((y + h) * sy) + pad)
    crop = _PIL_Image.fromarray(orig_rgb[y1:y2, x1:x2])
    crop.thumbnail((200, 200), _PIL_Image.BILINEAR)
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _scale_bbox(proc_bbox: list[int],
                orig_w: int, orig_h: int,
                proc_w: int, proc_h: int) -> list[int]:
    x, y, w, h = proc_bbox
    sx = orig_w / proc_w
    sy = orig_h / proc_h
    return [int(x * sx), int(y * sy), int(w * sx), int(h * sy)]


class LeftObjectDetector:
    """
    Per-camera left-object detector using a dual exponential-moving-average
    background subtraction approach.

    Parameters
    ----------
    dwell_s:
        Seconds a static foreground region must persist before an event is fired.
        Default 30.  Override with ``PCS_LEFT_OBJ_DWELL_S`` env var in the app.
    fg_thresh:
        Pixel intensity difference (0–255) to declare a pixel as foreground.
        Default 22.
    min_blob_frac:
        Minimum blob area as a fraction of the processed-frame area.
        Default 0.008 (≈ 0.8 % of 320×180 ≈ 461 px).
    stale_s:
        Seconds before an unmatched candidate is discarded.  Default 8.
    cooldown_s:
        Minimum gap between repeated events for the same camera.  Default 60.
    alpha_slow / alpha_fast:
        EMA learning rates for the two backgrounds.
    """

    def __init__(
        self,
        *,
        dwell_s:       float = 30.0,
        fg_thresh:     float = 22.0,
        min_blob_frac: float = 0.008,
        stale_s:       float = 8.0,
        cooldown_s:    float = 60.0,
        alpha_slow:    float = 0.002,
        alpha_fast:    float = 0.05,
        proc_w:        int   = 320,
        proc_h:        int   = 180,
    ) -> None:
        self.dwell_s       = dwell_s
        self.fg_thresh     = fg_thresh
        self.min_blob_frac = min_blob_frac
        self.stale_s       = stale_s
        self.cooldown_s    = cooldown_s
        self.alpha_slow    = alpha_slow
        self.alpha_fast    = alpha_fast
        self.proc_w        = proc_w
        self.proc_h        = proc_h
        self._states: dict[str, _CameraState] = {}
        self._states_lock = threading.Lock()

    def _get_state(self, camera_id: str) -> _CameraState:
        with self._states_lock:
            if camera_id not in self._states:
                self._states[camera_id] = _CameraState(
                    proc_w=self.proc_w,
                    proc_h=self.proc_h,
                )
            return self._states[camera_id]

    def reset(self, camera_id: str) -> None:
        """Discard all background models and candidates for a camera."""
        with self._states_lock:
            self._states.pop(camera_id, None)

    def process_frame(
        self, camera_id: str, frame_bytes: bytes
    ) -> list[dict]:
        """
        Process one JPEG frame for the given camera.

        Returns a (possibly empty) list of detection dicts, each containing:

        .. code-block:: python

            {
              "camera_id"     : str,
              "bbox"          : [x, y, w, h],  # original-frame pixels
              "first_seen_utc": str,            # ISO-8601 UTC
              "detected_utc"  : str,            # ISO-8601 UTC
              "dwell_s"       : float,
              "snapshot_b64"  : str,            # data-URI or ""
            }

        The method is thread-safe; each camera has its own lock.
        """
        state = self._get_state(camera_id)
        with state.lock:
            return self._run(state, camera_id, frame_bytes)

    def _run(
        self, state: _CameraState, camera_id: str, frame_bytes: bytes
    ) -> list[dict]:
        decoded = _decode_frame(frame_bytes, state.proc_w, state.proc_h)
        if decoded is None:
            return []
        gray_f, orig_rgb = decoded
        orig_h, orig_w = orig_rgb.shape[:2]

        # ── First frame: initialise both backgrounds and return ───────────────
        if state.bg_slow is None:
            state.bg_slow = gray_f.copy()
            state.bg_fast = gray_f.copy()
            return []

        # ── Foreground masks ──────────────────────────────────────────────────
        diff_slow = np.abs(gray_f - state.bg_slow)
        diff_fast = np.abs(gray_f - state.bg_fast)
        fg_slow = (diff_slow > self.fg_thresh).astype(np.uint8)
        fg_fast = (diff_fast > self.fg_thresh).astype(np.uint8)

        # Left-object mask: present vs slow bg AND not currently moving
        static_mask = fg_slow & (1 - fg_fast)

        # ── Update backgrounds ────────────────────────────────────────────────
        # fast_bg: always update — tracks short-term changes
        state.bg_fast = state.bg_fast * (1 - self.alpha_fast) + gray_f * self.alpha_fast
        # slow_bg: only update background pixels (freeze under detected objects)
        bg_pixels = (fg_slow == 0)
        state.bg_slow[bg_pixels] = (
            state.bg_slow[bg_pixels] * (1 - self.alpha_slow)
            + gray_f[bg_pixels] * self.alpha_slow
        )

        # ── Morphological cleanup ─────────────────────────────────────────────
        static_mask = _morpho_clean(static_mask)

        # ── Blob detection ────────────────────────────────────────────────────
        min_area = self.min_blob_frac * state.proc_w * state.proc_h
        raw_bboxes = _find_blobs(static_mask)
        current_bboxes = [
            bb for bb in raw_bboxes
            if bb[2] * bb[3] >= min_area
        ]

        now_s = time.monotonic()

        # ── Candidate matching by IoU ─────────────────────────────────────────
        matched_ids: set[int] = set()
        for bbox in current_bboxes:
            best_iou, best_id = 0.0, None
            for bid, cand in state.candidates.items():
                iou = _iou(bbox, cand.bbox)
                if iou > best_iou:
                    best_iou, best_id = iou, bid
            if best_id is not None and best_iou >= 0.25:
                state.candidates[best_id].bbox = bbox
                state.candidates[best_id].last_seen_s = now_s
                matched_ids.add(best_id)
            else:
                bid = state.next_blob_id
                state.next_blob_id += 1
                state.candidates[bid] = _BlobCandidate(
                    blob_id=bid,
                    bbox=bbox,
                    first_seen_s=now_s,
                    last_seen_s=now_s,
                )
                matched_ids.add(bid)

        # ── Prune stale candidates ────────────────────────────────────────────
        for bid in [k for k, c in state.candidates.items()
                    if k not in matched_ids and now_s - c.last_seen_s > self.stale_s]:
            del state.candidates[bid]

        # ── Fire events for candidates that crossed the dwell threshold ───────
        detections: list[dict] = []
        for cand in state.candidates.values():
            dwell = now_s - cand.first_seen_s
            if dwell < self.dwell_s:
                continue
            if cand.fired and now_s < cand.cooldown_until_s:
                continue
            # Fire
            cand.fired = True
            cand.cooldown_until_s = now_s + self.cooldown_s
            orig_bbox  = _scale_bbox(cand.bbox, orig_w, orig_h, state.proc_w, state.proc_h)
            snap_b64   = _crop_b64(orig_rgb, cand.bbox, state.proc_w, state.proc_h)
            first_utc  = _utcnow()   # approximate; caller may have a better timestamp
            detections.append({
                "camera_id":      camera_id,
                "bbox":           orig_bbox,
                "first_seen_utc": first_utc,
                "detected_utc":   _utcnow(),
                "dwell_s":        round(dwell, 1),
                "snapshot_b64":   snap_b64,
            })
            log.info(
                "[LOD] left_object camera=%s bbox=%s dwell=%.1fs",
                camera_id, orig_bbox, dwell,
            )

        return detections
