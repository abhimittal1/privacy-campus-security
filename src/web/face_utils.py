"""
face_utils.py – Multi-face detection and L2-normalised embedding extraction.

Public API
----------
::

    results = generate_embedding_from_image(image, model, device="cpu")

Each element of *results* is a :class:`FaceResult` TypedDict::

    {
        "bbox":          (x, y, w, h),   # pixel coords, clipped to image bounds
        "embedding":     np.ndarray,      # float32, shape (512,), L2-normalized
        "face_id_local": int,             # 0-based index within this frame
    }

Detector selection
------------------
1. **OpenCV DNN** (SSD/ResNet-Caffe) – used when both env vars are set:
   - ``PCS_FACE_PROTOTXT``   – path to ``deploy.prototxt``
   - ``PCS_FACE_CAFFEMODEL`` – path to ``res10_300x300_ssd_iter_140000.caffemodel``

   The OpenCV model files can be obtained from the opencv_extra repository:
   https://github.com/opencv/opencv_extra/tree/master/testdata/dnn

2. **Haar cascade** (``haarcascade_frontalface_default.xml``, bundled with
   *opencv-python*) – used when DNN weights are unavailable.

Non-max suppression
-------------------
IoU-based greedy NMS is applied after detection (threshold configurable via
:data:`NMS_IOU_THRESH`).  This collapses overlapping candidate boxes for the
same face before cropping and embedding.

NumPy ↔ torch bridge
---------------------
All tensor construction uses the same bridge-free path as ``flask_app.py``:
PIL → ``tobytes()`` → ``bytearray`` → ``torch.frombuffer()``.  This avoids
the ``torch.from_numpy`` / ``tensor.numpy()`` C-API paths that are broken
under NumPy ≥ 2.0.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence, TypedDict, Union

import cv2
import numpy as np
import torch
from PIL import Image

from src.models.backbone import BackboneEmbedder

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Confidence threshold for the DNN detector (0.0–1.0).
DNN_CONFIDENCE_THRESH: float = 0.5

#: Input blob size expected by the OpenCV SSD face model.
DNN_INPUT_SIZE: tuple[int, int] = (300, 300)

#: Scale factor used by the DNN blob pre-processing step.
DNN_SCALE_FACTOR: float = 1.0

#: Mean pixel subtraction values (BGR) for the DNN model.
DNN_MEAN: tuple[float, float, float] = (104.0, 177.0, 123.0)

#: IoU threshold above which a lower-scoring box is suppressed.
NMS_IOU_THRESH: float = 0.35

#: Haar cascade detector parameters.
HAAR_SCALE_FACTOR: float = 1.1
HAAR_MIN_NEIGHBOURS: int = 5
HAAR_MIN_FACE_PX: int = 30  # minimum face dimension (pixels)

#: Image size fed into the backbone (must match lfw_loader.DEFAULT_TRANSFORM).
_EMBED_IMG_SIZE: int = 112

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class FaceResult(TypedDict):
    """Single detected face with its bounding box and embedding vector."""

    bbox: tuple[int, int, int, int]  # (x, y, w, h) in pixel coords
    embedding: np.ndarray            # float32, shape (embedding_dim,), L2-norm=1
    face_id_local: int               # 0-based index within this frame


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    """
    Compute intersection-over-union between two (x, y, w, h) boxes.

    Returns a value in [0.0, 1.0].
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ix = max(ax, bx)
    iy = max(ay, by)
    ix2 = min(ax + aw, bx + bw)
    iy2 = min(ay + ah, by + bh)

    inter_w = max(0, ix2 - ix)
    inter_h = max(0, iy2 - iy)
    inter = inter_w * inter_h

    if inter == 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _nms(
    boxes: list[tuple[int, int, int, int]],
    scores: list[float],
    iou_thresh: float = NMS_IOU_THRESH,
) -> list[int]:
    """
    Greedy non-max suppression.

    Args:
        boxes:      List of ``(x, y, w, h)`` bounding boxes.
        scores:     Confidence score for each box (higher = more confident).
        iou_thresh: Boxes whose IoU with a kept box exceeds this value are suppressed.

    Returns:
        Sorted list of *kept* indices into *boxes*.
    """
    if not boxes:
        return []

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept: list[int] = []
    suppressed = [False] * len(order)

    for i, idx in enumerate(order):
        if suppressed[i]:
            continue
        kept.append(idx)
        for j in range(i + 1, len(order)):
            if not suppressed[j]:
                if _iou(boxes[idx], boxes[order[j]]) > iou_thresh:
                    suppressed[j] = True

    return sorted(kept)


def _pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """
    Bridge-free PIL → float32 ``(C, H, W)`` tensor.

    Resizes the image to ``_EMBED_IMG_SIZE × _EMBED_IMG_SIZE``, converts
    pixel values to ``[-1, 1]``, and packs into a ``(C, H, W)`` tensor
    without ever touching the NumPy C-API bridge.
    """
    pil_img = pil_img.resize((_EMBED_IMG_SIZE, _EMBED_IMG_SIZE), Image.BILINEAR)
    raw = bytearray(pil_img.tobytes())                          # H*W*3 uint8
    t = torch.frombuffer(raw, dtype=torch.uint8).clone()        # 1-D uint8
    t = t.reshape(_EMBED_IMG_SIZE, _EMBED_IMG_SIZE, 3)          # (H, W, C)
    t = t.float() / 255.0                                       # [0, 1]
    t = (t - 0.5) / 0.5                                         # [-1, 1]
    return t.permute(2, 0, 1)                                   # (C, H, W)


def _to_pil(image: Union[Image.Image, np.ndarray, str, Path]) -> Image.Image:
    """
    Normalise various image representations to a PIL RGB image.

    Accepted types:
    - ``PIL.Image.Image`` — returned as-is (converted to RGB).
    - ``numpy.ndarray`` — shape ``(H, W, 3)`` or ``(H, W)``, uint8 or float32.
      Float arrays are assumed to be ``[0, 1]`` and multiplied by 255.
    - ``str`` / ``pathlib.Path`` — file path loaded with PIL.
    """
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        return Image.fromarray(image[..., :3], mode="RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


# ---------------------------------------------------------------------------
# Detector singleton
# ---------------------------------------------------------------------------

# Module-level cache: (_detector, _detector_type)
_detector_cache: tuple[object, str] | None = None


def _load_detector() -> tuple[object, str]:
    """
    Load and cache the face detector.

    Returns
    -------
    (detector, kind)  where ``kind`` is ``"dnn"`` or ``"haar"``.
    """
    global _detector_cache  # noqa: PLW0603
    if _detector_cache is not None:
        return _detector_cache

    proto = os.environ.get("PCS_FACE_PROTOTXT", "")
    caffe = os.environ.get("PCS_FACE_CAFFEMODEL", "")

    if proto and caffe and Path(proto).is_file() and Path(caffe).is_file():
        try:
            net = cv2.dnn.readNetFromCaffe(proto, caffe)
            log.info("Face detector: OpenCV DNN loaded from %s / %s", proto, caffe)
            _detector_cache = (net, "dnn")
            return _detector_cache
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to load DNN face detector (%s), falling back to Haar.", exc)

    # Fallback: Haar cascade (always available with opencv-python)
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(
            f"Haar cascade could not be loaded from {cascade_path}. "
            "Ensure opencv-python is correctly installed."
        )
    log.info("Face detector: Haar cascade loaded from %s", cascade_path)
    _detector_cache = (cascade, "haar")
    return _detector_cache


def _clear_detector_cache() -> None:
    """Reset the cached detector (used in tests to force reloading)."""
    global _detector_cache  # noqa: PLW0603
    _detector_cache = None


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _detect_dnn(
    net: cv2.dnn_Net,  # type: ignore[name-defined]
    rgb_array: np.ndarray,
    conf_thresh: float = DNN_CONFIDENCE_THRESH,
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """
    Run the OpenCV SSD/ResNet Caffe face detector on a uint8 RGB image.

    Returns
    -------
    (boxes, scores)  where each box is ``(x, y, w, h)`` in pixel coords.
    """
    h, w = rgb_array.shape[:2]
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    blob = cv2.dnn.blobFromImage(
        bgr,
        scalefactor=DNN_SCALE_FACTOR,
        size=DNN_INPUT_SIZE,
        mean=DNN_MEAN,
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    detections = net.forward()  # shape: (1, 1, N, 7)

    boxes: list[tuple[int, int, int, int]] = []
    scores: list[float] = []

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf_thresh:
            continue
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)
        # Clip to image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw > 0 and bh > 0:
            boxes.append((x1, y1, bw, bh))
            scores.append(confidence)

    return boxes, scores


def _detect_haar(
    cascade: cv2.CascadeClassifier,
    rgb_array: np.ndarray,
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """
    Run the Haar cascade face detector on a uint8 RGB image.

    Returns
    -------
    (boxes, scores)  where each box is ``(x, y, w, h)`` in pixel coords.
    Haar cascade does not produce confidence scores, so all scores are set to 1.0.
    """
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=HAAR_SCALE_FACTOR,
        minNeighbors=HAAR_MIN_NEIGHBOURS,
        minSize=(HAAR_MIN_FACE_PX, HAAR_MIN_FACE_PX),
    )

    if len(faces) == 0:
        return [], []

    boxes = [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
    scores = [1.0] * len(boxes)
    return boxes, scores


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_embedding_from_image(
    image: Union[Image.Image, np.ndarray, str, Path],
    model: Optional[BackboneEmbedder] = None,
    device: str = "cpu",
    *,
    conf_thresh: float = DNN_CONFIDENCE_THRESH,
    nms_iou_thresh: float = NMS_IOU_THRESH,
) -> list[FaceResult]:
    """
    Detect all faces in *image* and return their L2-normalised embeddings.

    Processing pipeline
    -------------------
    1. Normalise *image* to a PIL RGB image.
    2. Run the cached face detector (DNN → Haar fallback).
    3. Apply IoU-based non-max suppression.
    4. Crop each surviving box, apply bridge-free preprocessing.
    5. Batch-embed all faces through the backbone in a single forward pass.
    6. Return one :class:`FaceResult` per detected face.

    Parameters
    ----------
    image:
        Input frame.  Accepts a ``PIL.Image``, a uint8/float32 ``numpy``
        array with shape ``(H, W, 3)`` or ``(H, W)``, or a file path.
    model:
        A :class:`~src.models.backbone.BackboneEmbedder` in eval mode.
        If ``None`` a new CPU instance is created automatically (slow —
        prefer passing a pre-loaded model from the Flask app).
    device:
        PyTorch device string (``"cpu"`` or ``"cuda"``).
    conf_thresh:
        Minimum DNN confidence score (ignored for Haar cascade).
    nms_iou_thresh:
        IoU threshold for suppressing overlapping detections.

    Returns
    -------
    list[FaceResult]
        One entry per detected face, ordered by ``face_id_local`` (0-based).
        Returns an empty list if no faces are detected.
    """
    # ── 0. Prepare model ────────────────────────────────────────────────────────
    if model is None:
        log.warning(
            "generate_embedding_from_image: no model supplied, loading default "
            "BackboneEmbedder (pretrained=False).  Pass a pre-loaded model for "
            "production use."
        )
        model = BackboneEmbedder(pretrained=False).eval()

    torch_device = torch.device(device)
    model = model.to(torch_device).eval()

    # ── 1. Normalise input ──────────────────────────────────────────────────────
    pil_img = _to_pil(image)
    rgb_array = np.array(pil_img, dtype=np.uint8)   # (H, W, 3) uint8

    img_h, img_w = rgb_array.shape[:2]

    # ── 2. Detect ───────────────────────────────────────────────────────────────
    detector, kind = _load_detector()

    if kind == "dnn":
        raw_boxes, raw_scores = _detect_dnn(
            detector,  # type: ignore[arg-type]
            rgb_array,
            conf_thresh=conf_thresh,
        )
    else:  # "haar"
        raw_boxes, raw_scores = _detect_haar(
            detector,  # type: ignore[arg-type]
            rgb_array,
        )

    if not raw_boxes:
        log.debug("generate_embedding_from_image: no faces detected in frame.")
        return []

    # ── 3. Non-max suppression ──────────────────────────────────────────────────
    kept_idxs = _nms(raw_boxes, raw_scores, iou_thresh=nms_iou_thresh)
    boxes = [raw_boxes[i] for i in kept_idxs]

    log.debug(
        "generate_embedding_from_image: detector=%s detected=%d after_nms=%d",
        kind,
        len(raw_boxes),
        len(boxes),
    )

    # ── 4. Crop and preprocess each face ────────────────────────────────────────
    face_tensors: list[torch.Tensor] = []
    clipped_boxes: list[tuple[int, int, int, int]] = []

    for x, y, w, h in boxes:
        # Clip to image bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(img_w, x + w)
        y2 = min(img_h, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = pil_img.crop((x1, y1, x2, y2))          # PIL.Image crop
        face_tensors.append(_pil_to_tensor(crop))       # (C, H, W)
        clipped_boxes.append((x1, y1, x2 - x1, y2 - y1))

    if not face_tensors:
        return []

    # ── 5. Batch forward pass ───────────────────────────────────────────────────
    batch = torch.stack(face_tensors).to(torch_device)  # (N, C, H, W)
    with torch.no_grad():
        embeddings_t = model(batch)                      # (N, 512), L2-normed

    # Bridge-free numpy conversion: tolist() → Python floats → np.array()
    embeddings_np = [
        np.array(embeddings_t[i].tolist(), dtype=np.float32)
        for i in range(embeddings_t.shape[0])
    ]

    # ── 6. Build results ────────────────────────────────────────────────────────
    results: list[FaceResult] = [
        FaceResult(
            bbox=clipped_boxes[i],
            embedding=embeddings_np[i],
            face_id_local=i,
        )
        for i in range(len(embeddings_np))
    ]

    return results
