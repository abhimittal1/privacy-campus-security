"""
test_face_utils.py – Tests for src/web/face_utils.py.

Strategy
--------
- The backbone is instantiated with ``pretrained=False`` so tests run without
  network access.
- Face detector calls are monkeypatched to return deterministic bounding boxes,
  so tests do not depend on image content or model accuracy.
- Pure-logic helpers (_iou, _nms, _to_pil, _pil_to_tensor) are tested with
  synthetic inputs.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.backbone import BackboneEmbedder
from src.web.face_utils import (
    FaceResult,
    _clear_detector_cache,
    _iou,
    _nms,
    _pil_to_tensor,
    _to_pil,
    generate_embedding_from_image,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_model() -> BackboneEmbedder:
    """Random-weight BackboneEmbedder (no network access, CPU only)."""
    return BackboneEmbedder(pretrained=False).eval()


@pytest.fixture()
def blank_image() -> Image.Image:
    """300×300 solid-grey PIL RGB image used as a dummy frame."""
    return Image.new("RGB", (300, 300), color=(120, 120, 120))


@pytest.fixture(autouse=True)
def clear_detector_cache():
    """
    Reset the module-level detector singleton before and after every test so
    that environment-variable patches and monkeypatches take effect cleanly.
    """
    _clear_detector_cache()
    yield
    _clear_detector_cache()


# ---------------------------------------------------------------------------
# Helper: make a fake Haar cascade detector mock that never loads any file
# ---------------------------------------------------------------------------


def _mock_cascade(face_array: np.ndarray) -> MagicMock:
    """
    Return a :class:`~unittest.mock.MagicMock` that mimics
    ``cv2.CascadeClassifier`` and returns *face_array* from ``detectMultiScale``.
    """
    cascade = MagicMock()
    cascade.empty.return_value = False
    cascade.detectMultiScale.return_value = face_array
    return cascade


# ---------------------------------------------------------------------------
# _iou
# ---------------------------------------------------------------------------


class TestIoU:
    def test_identical_boxes_give_one(self) -> None:
        assert _iou((0, 0, 100, 100), (0, 0, 100, 100)) == pytest.approx(1.0)

    def test_non_overlapping_boxes_give_zero(self) -> None:
        assert _iou((0, 0, 50, 50), (100, 100, 50, 50)) == pytest.approx(0.0)

    def test_half_overlap_symmetric(self) -> None:
        # Two boxes sharing exactly half their area each
        iou_ab = _iou((0, 0, 100, 100), (50, 0, 100, 100))
        iou_ba = _iou((50, 0, 100, 100), (0, 0, 100, 100))
        assert iou_ab == pytest.approx(iou_ba)
        assert 0.0 < iou_ab < 1.0

    def test_contained_box(self) -> None:
        # Inner box fully inside outer box → IoU = inner_area / outer_area
        iou = _iou((0, 0, 100, 100), (25, 25, 50, 50))
        # inner_area=2500, outer_area=10000, union=10000, inter=2500
        assert iou == pytest.approx(2500 / 10000)

    def test_zero_size_box(self) -> None:
        assert _iou((0, 0, 0, 0), (0, 0, 100, 100)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _nms
# ---------------------------------------------------------------------------


class TestNMS:
    def test_empty_input(self) -> None:
        assert _nms([], []) == []

    def test_single_box_is_kept(self) -> None:
        assert _nms([(0, 0, 100, 100)], [0.9]) == [0]

    def test_non_overlapping_boxes_all_kept(self) -> None:
        boxes = [(0, 0, 50, 50), (100, 0, 50, 50), (200, 0, 50, 50)]
        scores = [0.9, 0.8, 0.7]
        kept = _nms(boxes, scores)
        assert set(kept) == {0, 1, 2}

    def test_heavily_overlapping_keeps_highest_score(self) -> None:
        # Two nearly identical boxes — lower-score one should be suppressed.
        boxes = [(0, 0, 100, 100), (5, 5, 95, 95)]
        scores = [0.9, 0.5]
        kept = _nms(boxes, scores, iou_thresh=0.3)
        assert 0 in kept
        assert len(kept) == 1

    def test_nms_returns_sorted_indices(self) -> None:
        boxes = [(200, 0, 50, 50), (0, 0, 50, 50), (100, 0, 50, 50)]
        scores = [0.7, 0.9, 0.8]
        kept = _nms(boxes, scores)
        assert kept == sorted(kept)

    def test_three_faces_after_nms(self) -> None:
        # Three non-overlapping boxes of decreasing confidence.
        boxes = [(0, 0, 60, 60), (80, 0, 60, 60), (160, 0, 60, 60)]
        scores = [0.95, 0.85, 0.75]
        kept = _nms(boxes, scores, iou_thresh=0.3)
        assert len(kept) == 3


# ---------------------------------------------------------------------------
# _to_pil
# ---------------------------------------------------------------------------


class TestToPil:
    def test_pil_input_passthrough(self, blank_image: Image.Image) -> None:
        pil = _to_pil(blank_image)
        assert isinstance(pil, Image.Image)
        assert pil.mode == "RGB"

    def test_numpy_uint8_input(self) -> None:
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        pil = _to_pil(arr)
        assert isinstance(pil, Image.Image)
        assert pil.mode == "RGB"

    def test_numpy_float32_input(self) -> None:
        arr = np.full((100, 100, 3), 0.5, dtype=np.float32)
        pil = _to_pil(arr)
        assert isinstance(pil, Image.Image)

    def test_grayscale_numpy_expands_to_rgb(self) -> None:
        arr = np.zeros((100, 100), dtype=np.uint8)
        pil = _to_pil(arr)
        assert pil.mode == "RGB"
        assert pil.size == (100, 100)

    def test_file_path_as_str(self, tmp_path: Path, blank_image: Image.Image) -> None:
        p = tmp_path / "test.jpg"
        blank_image.save(str(p))
        pil = _to_pil(str(p))
        assert isinstance(pil, Image.Image)
        assert pil.mode == "RGB"

    def test_file_path_as_pathlib(self, tmp_path: Path, blank_image: Image.Image) -> None:
        p = tmp_path / "test.png"
        blank_image.save(p)
        pil = _to_pil(p)
        assert isinstance(pil, Image.Image)

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported image type"):
            _to_pil(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _pil_to_tensor
# ---------------------------------------------------------------------------


class TestPilToTensor:
    def test_output_shape(self, blank_image: Image.Image) -> None:
        t = _pil_to_tensor(blank_image)
        assert t.shape == (3, 112, 112)

    def test_output_dtype(self, blank_image: Image.Image) -> None:
        t = _pil_to_tensor(blank_image)
        assert t.dtype == torch.float32

    def test_values_in_range(self, blank_image: Image.Image) -> None:
        t = _pil_to_tensor(blank_image)
        assert t.min().item() >= -1.0
        assert t.max().item() <= 1.0

    def test_black_image_maps_to_minus_one(self) -> None:
        black = Image.new("RGB", (50, 50), (0, 0, 0))
        t = _pil_to_tensor(black)
        assert torch.allclose(t, torch.full_like(t, -1.0), atol=1e-4)

    def test_white_image_maps_to_plus_one(self) -> None:
        white = Image.new("RGB", (50, 50), (255, 255, 255))
        t = _pil_to_tensor(white)
        assert torch.allclose(t, torch.ones_like(t), atol=1e-4)


# ---------------------------------------------------------------------------
# generate_embedding_from_image – integration-style tests with mocked detector
# ---------------------------------------------------------------------------


def _patch_haar(bboxes: np.ndarray):
    """
    Context manager that patches the detector singleton with a Haar mock that
    returns *bboxes* from detectMultiScale.
    """
    import src.web.face_utils as fu

    mock_cascade = _mock_cascade(bboxes)

    return patch.object(
        fu,
        "_detector_cache",
        new=(mock_cascade, "haar"),
        create=True,
    )


# Use parametrize over a helper instead of trying to nest patch.object
# ─────────────────────────────────────────────────────────────────────


@pytest.fixture()
def two_face_bboxes() -> np.ndarray:
    """Two non-overlapping bounding boxes that fit inside a 300×300 image."""
    return np.array([[10, 10, 80, 80], [170, 10, 80, 80]])


@pytest.fixture()
def three_face_bboxes() -> np.ndarray:
    """Three non-overlapping bounding boxes."""
    return np.array([[10, 10, 70, 70], [110, 10, 70, 70], [210, 10, 70, 70]])


class TestGenerateEmbedding:
    # ── Multi-face assertions ─────────────────────────────────────────────────

    def test_two_faces_returns_two_results(
        self, blank_image: Image.Image, tiny_model: BackboneEmbedder, two_face_bboxes: np.ndarray
    ) -> None:
        """Core requirement: two detected faces → two FaceResult entries."""
        import src.web.face_utils as fu

        with patch.object(fu, "_detector_cache", ((_mock_cascade(two_face_bboxes)), "haar")):
            results = generate_embedding_from_image(blank_image, tiny_model)

        assert len(results) > 1, "Expected more than one embedding for two faces"
        assert len(results) == 2

    def test_three_faces_returns_three_results(
        self,
        blank_image: Image.Image,
        tiny_model: BackboneEmbedder,
        three_face_bboxes: np.ndarray,
    ) -> None:
        """Three detected faces → three FaceResult entries."""
        import src.web.face_utils as fu

        with patch.object(fu, "_detector_cache", (_mock_cascade(three_face_bboxes), "haar")):
            results = generate_embedding_from_image(blank_image, tiny_model)

        assert len(results) == 3

    def test_multi_face_returns_more_than_one_embedding(
        self, blank_image: Image.Image, tiny_model: BackboneEmbedder, two_face_bboxes: np.ndarray
    ) -> None:
        """Explicit assertion matching the spec: multi-face → >1 embeddings."""
        import src.web.face_utils as fu

        with patch.object(fu, "_detector_cache", (_mock_cascade(two_face_bboxes), "haar")):
            results = generate_embedding_from_image(blank_image, tiny_model)

        assert len(results) > 1

    # ── FaceResult structure ──────────────────────────────────────────────────

    def test_result_has_required_fields(
        self, blank_image: Image.Image, tiny_model: BackboneEmbedder, two_face_bboxes: np.ndarray
    ) -> None:
        import src.web.face_utils as fu

        with patch.object(fu, "_detector_cache", (_mock_cascade(two_face_bboxes), "haar")):
            results = generate_embedding_from_image(blank_image, tiny_model)

        for r in results:
            assert "bbox" in r
            assert "embedding" in r
            assert "face_id_local" in r

    def test_bbox_is_four_int_tuple(
        self, blank_image: Image.Image, tiny_model: BackboneEmbedder, two_face_bboxes: np.ndarray
    ) -> None:
        import src.web.face_utils as fu

        with patch.object(fu, "_detector_cache", (_mock_cascade(two_face_bboxes), "haar")):
            results = generate_embedding_from_image(blank_image, tiny_model)

        for r in results:
            bbox = r["bbox"]
            assert isinstance(bbox, tuple)
            assert len(bbox) == 4
            assert all(isinstance(v, int) for v in bbox)

    def test_embedding_shape_is_512(
        self, blank_image: Image.Image, tiny_model: BackboneEmbedder, two_face_bboxes: np.ndarray
    ) -> None:
        import src.web.face_utils as fu

        with patch.object(fu, "_detector_cache", (_mock_cascade(two_face_bboxes), "haar")):
            results = generate_embedding_from_image(blank_image, tiny_model)

        for r in results:
            assert r["embedding"].shape == (512,)
            assert r["embedding"].dtype == np.float32

    def test_embeddings_are_l2_normalized(
        self, blank_image: Image.Image, tiny_model: BackboneEmbedder, two_face_bboxes: np.ndarray
    ) -> None:
        """All embeddings must have unit L2 norm (the backbone's final F.normalize)."""
        import src.web.face_utils as fu

        with patch.object(fu, "_detector_cache", (_mock_cascade(two_face_bboxes), "haar")):
            results = generate_embedding_from_image(blank_image, tiny_model)

        for r in results:
            norm = float(np.linalg.norm(r["embedding"]))
            assert norm == pytest.approx(1.0, abs=1e-5), f"Embedding norm={norm} ≠ 1"

    def test_face_id_local_is_sequential_from_zero(
        self, blank_image: Image.Image, tiny_model: BackboneEmbedder, three_face_bboxes: np.ndarray
    ) -> None:
        import src.web.face_utils as fu

        with patch.object(fu, "_detector_cache", (_mock_cascade(three_face_bboxes), "haar")):
            results = generate_embedding_from_image(blank_image, tiny_model)

        ids = [r["face_id_local"] for r in results]
        assert ids == list(range(len(results)))

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_no_faces_returns_empty_list(
        self, blank_image: Image.Image, tiny_model: BackboneEmbedder
    ) -> None:
        import src.web.face_utils as fu

        with patch.object(fu, "_detector_cache", (_mock_cascade(np.array([])), "haar")):
            results = generate_embedding_from_image(blank_image, tiny_model)

        assert results == []

    def test_single_face_returns_one_result(
        self, blank_image: Image.Image, tiny_model: BackboneEmbedder
    ) -> None:
        import src.web.face_utils as fu

        single_bbox = np.array([[50, 50, 80, 80]])
        with patch.object(fu, "_detector_cache", (_mock_cascade(single_bbox), "haar")):
            results = generate_embedding_from_image(blank_image, tiny_model)

        assert len(results) == 1
        assert results[0]["face_id_local"] == 0

    # ── Input type acceptance ─────────────────────────────────────────────────

    def test_accepts_numpy_array(
        self, tiny_model: BackboneEmbedder, two_face_bboxes: np.ndarray
    ) -> None:
        import src.web.face_utils as fu

        arr = np.zeros((300, 300, 3), dtype=np.uint8)
        with patch.object(fu, "_detector_cache", (_mock_cascade(two_face_bboxes), "haar")):
            results = generate_embedding_from_image(arr, tiny_model)

        assert len(results) == 2

    def test_accepts_file_path(
        self,
        tmp_path: Path,
        blank_image: Image.Image,
        tiny_model: BackboneEmbedder,
        two_face_bboxes: np.ndarray,
    ) -> None:
        import src.web.face_utils as fu

        p = tmp_path / "frame.jpg"
        blank_image.save(str(p))

        with patch.object(fu, "_detector_cache", (_mock_cascade(two_face_bboxes), "haar")):
            results = generate_embedding_from_image(str(p), tiny_model)

        assert len(results) == 2

    # ── NMS de-duplication ────────────────────────────────────────────────────

    def test_nms_collapses_heavily_overlapping_boxes(
        self, blank_image: Image.Image, tiny_model: BackboneEmbedder
    ) -> None:
        """Four boxes — two pairs of heavy overlaps — reduces to two faces after NMS."""
        import src.web.face_utils as fu

        # Two clusters: cluster A (left) and cluster B (right), each with two
        # nearly-identical boxes that should be merged by NMS.
        overlapping = np.array([
            [10, 10, 80, 80],   # cluster A, box 1
            [12, 12, 80, 80],   # cluster A, box 2 (high overlap with box 1)
            [180, 10, 80, 80],  # cluster B, box 1
            [182, 12, 80, 80],  # cluster B, box 2 (high overlap with cluster B box 1)
        ])

        with patch.object(fu, "_detector_cache", (_mock_cascade(overlapping), "haar")):
            results = generate_embedding_from_image(blank_image, tiny_model)

        assert len(results) == 2, (
            f"NMS should collapse 4 overlapping boxes into 2 faces, got {len(results)}"
        )

    # ── DNN detector path ─────────────────────────────────────────────────────

    def test_dnn_detector_path(
        self, blank_image: Image.Image, tiny_model: BackboneEmbedder
    ) -> None:
        """Ensure the function works when the 'dnn' detector path is active."""
        import src.web.face_utils as fu

        # Simulate two detections from the DNN path
        boxes = [(10, 10, 80, 80), (170, 10, 80, 80)]
        scores = [0.97, 0.92]

        with patch("src.web.face_utils._detect_dnn", return_value=(boxes, scores)):
            # _detector_cache must hold a 'dnn' entry so the dnn branch is chosen
            fake_net = MagicMock()
            with patch.object(fu, "_detector_cache", (fake_net, "dnn")):
                results = generate_embedding_from_image(blank_image, tiny_model)

        assert len(results) == 2
        assert all(r["embedding"].shape == (512,) for r in results)

    # ── Detector loading ──────────────────────────────────────────────────────

    def test_haar_detector_loaded_when_dnn_env_vars_absent(self) -> None:
        """Without PCS_FACE_PROTOTXT/CAFFEMODEL set, the Haar cascade must load."""
        import src.web.face_utils as fu

        env = {k: v for k, v in os.environ.items()
               if k not in ("PCS_FACE_PROTOTXT", "PCS_FACE_CAFFEMODEL")}

        with patch.dict(os.environ, env, clear=True):
            _, kind = fu._load_detector()

        assert kind == "haar"

    def test_dnn_detector_loaded_when_env_vars_set(self, tmp_path: Path) -> None:
        """With valid env var paths, the DNN loader is attempted."""
        import src.web.face_utils as fu

        # Create dummy files (DNN loader will run but fail gracefully on bad content)
        proto = tmp_path / "deploy.prototxt"
        caffe = tmp_path / "model.caffemodel"
        proto.write_text("dummy")
        caffe.write_bytes(b"\x00" * 8)

        env_patch = {
            "PCS_FACE_PROTOTXT": str(proto),
            "PCS_FACE_CAFFEMODEL": str(caffe),
        }

        # Patch cv2.dnn.readNetFromCaffe to avoid actually parsing dummy files
        mock_net = MagicMock()
        with patch.dict(os.environ, env_patch):
            with patch("cv2.dnn.readNetFromCaffe", return_value=mock_net):
                _, kind = fu._load_detector()

        assert kind == "dnn"
