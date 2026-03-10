"""test_backbone.py – Unit tests for BackboneEmbedder, ProjectionHead, and embedding_utils."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from src.models.backbone import (
    EMBEDDING_DIM,
    BackboneEmbedder,
    ProjectionHead,
    _RESNET50_POOL_DIM,
)
from src.models.embedding_utils import save_embeddings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model() -> BackboneEmbedder:
    """Untrained model (pretrained=False) so tests run without network access."""
    return BackboneEmbedder(
        embedding_dim=EMBEDDING_DIM,
        freeze_base=True,
        pretrained=False,
    ).eval()


@pytest.fixture(scope="module")
def frozen_model() -> BackboneEmbedder:
    """Model with freeze_base=True for gradient-related assertions."""
    return BackboneEmbedder(freeze_base=True, pretrained=False).eval()


@pytest.fixture(scope="module")
def unfrozen_model() -> BackboneEmbedder:
    """Model with freeze_base=False for trainable-body assertions."""
    return BackboneEmbedder(freeze_base=False, pretrained=False).eval()


# ---------------------------------------------------------------------------
# BackboneEmbedder – shape & normalisation
# ---------------------------------------------------------------------------


class TestBackboneEmbedder:
    def test_output_shape_batch(self, model: BackboneEmbedder) -> None:
        """Embeddings must have shape (batch, EMBEDDING_DIM)."""
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, EMBEDDING_DIM)

    def test_output_shape_small_input(self, model: BackboneEmbedder) -> None:
        """ArcFace-style 112×112 input must also produce (B, 512)."""
        x = torch.randn(2, 3, 112, 112)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, EMBEDDING_DIM)

    def test_l2_normalised(self, model: BackboneEmbedder) -> None:
        """Every embedding row must have unit L2 norm."""
        x = torch.randn(8, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(8), atol=1e-5)

    def test_single_image(self, model: BackboneEmbedder) -> None:
        """Batch size of 1 must not break BatchNorm (eval mode)."""
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, EMBEDDING_DIM)

    def test_embedding_dim_attribute(self, model: BackboneEmbedder) -> None:
        """model.embedding_dim must match EMBEDDING_DIM constant."""
        assert model.embedding_dim == EMBEDDING_DIM

    def test_custom_embedding_dim(self) -> None:
        """Custom embedding_dim must propagate to output shape."""
        m = BackboneEmbedder(embedding_dim=256, pretrained=False).eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (2, 256)


# ---------------------------------------------------------------------------
# Freeze / unfreeze behaviour
# ---------------------------------------------------------------------------


class TestFreezeBase:
    def test_body_frozen(self, frozen_model: BackboneEmbedder) -> None:
        """freeze_base=True must set requires_grad=False on all body params."""
        for param in frozen_model.body.parameters():
            assert not param.requires_grad

    def test_projector_trainable_when_frozen(self, frozen_model: BackboneEmbedder) -> None:
        """Projection head must remain trainable even when body is frozen."""
        for param in frozen_model.projector.parameters():
            assert param.requires_grad

    def test_body_unfrozen(self, unfrozen_model: BackboneEmbedder) -> None:
        """freeze_base=False must leave all body params trainable."""
        for param in unfrozen_model.body.parameters():
            assert param.requires_grad


# ---------------------------------------------------------------------------
# ProjectionHead
# ---------------------------------------------------------------------------


class TestProjectionHead:
    def test_output_shape(self) -> None:
        """ProjectionHead must map (B, in_features, 1, 1) → (B, out_features)."""
        head = ProjectionHead(in_features=2048, hidden_dim=2048, out_features=512).eval()
        x = torch.randn(4, 2048, 1, 1)
        with torch.no_grad():
            out = head(x)
        assert out.shape == (4, 512)

    def test_custom_dims(self) -> None:
        """Smaller hidden_dim and out_features must be accepted."""
        head = ProjectionHead(in_features=512, hidden_dim=256, out_features=128).eval()
        x = torch.randn(3, 512, 1, 1)
        with torch.no_grad():
            out = head(x)
        assert out.shape == (3, 128)


# ---------------------------------------------------------------------------
# ArcFace weight loader stub
# ---------------------------------------------------------------------------


class TestLoadArcfaceWeights:
    def test_none_path_warns(self) -> None:
        """Calling load_arcface_weights(None) must emit a UserWarning."""
        m = BackboneEmbedder(pretrained=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m.load_arcface_weights(None)
        assert any(issubclass(w.category, UserWarning) for w in caught)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A non-existent path must raise FileNotFoundError."""
        m = BackboneEmbedder(pretrained=False)
        with pytest.raises(FileNotFoundError):
            m.load_arcface_weights(str(tmp_path / "nonexistent.pth"))


# ---------------------------------------------------------------------------
# embedding_utils.save_embeddings
# ---------------------------------------------------------------------------


class TestSaveEmbeddings:
    @pytest.fixture()
    def tiny_dataset(self) -> TensorDataset:
        """10 random 'images' with integer labels."""
        images = torch.randn(10, 3, 224, 224)
        labels = torch.arange(10)
        return TensorDataset(images, labels)

    def test_saves_file(self, tiny_dataset: TensorDataset, tmp_path: Path) -> None:
        """save_embeddings must create the output .pt file."""
        m = BackboneEmbedder(pretrained=False).eval()
        out = tmp_path / "emb.pt"
        save_embeddings(tiny_dataset, m, out, batch_size=4)
        assert out.exists()

    def test_embedding_shape(self, tiny_dataset: TensorDataset, tmp_path: Path) -> None:
        """Returned dict must contain embeddings of shape (N, 512)."""
        m = BackboneEmbedder(pretrained=False).eval()
        out = tmp_path / "emb.pt"
        payload = save_embeddings(tiny_dataset, m, out, batch_size=4)
        assert payload["embeddings"].shape == (10, EMBEDDING_DIM)

    def test_labels_shape(self, tiny_dataset: TensorDataset, tmp_path: Path) -> None:
        """Returned dict must contain labels tensor of shape (N,)."""
        m = BackboneEmbedder(pretrained=False).eval()
        out = tmp_path / "emb.pt"
        payload = save_embeddings(tiny_dataset, m, out, batch_size=4)
        assert payload["labels"].shape == (10,)

    def test_embeddings_l2_normalised(self, tiny_dataset: TensorDataset, tmp_path: Path) -> None:
        """All saved embeddings must have unit L2 norm."""
        m = BackboneEmbedder(pretrained=False).eval()
        out = tmp_path / "emb.pt"
        payload = save_embeddings(tiny_dataset, m, out, batch_size=4)
        norms = payload["embeddings"].norm(dim=1)
        assert torch.allclose(norms, torch.ones(10), atol=1e-5)

    def test_roundtrip_load(self, tiny_dataset: TensorDataset, tmp_path: Path) -> None:
        """Loading the saved .pt must reproduce the same embedding tensor."""
        m = BackboneEmbedder(pretrained=False).eval()
        out = tmp_path / "emb.pt"
        payload = save_embeddings(tiny_dataset, m, out, batch_size=4)
        loaded = torch.load(out, map_location="cpu")
        assert torch.allclose(payload["embeddings"], loaded["embeddings"])

    def test_empty_dataset_raises(self, tmp_path: Path) -> None:
        """An empty dataset must raise ValueError."""
        m = BackboneEmbedder(pretrained=False).eval()
        empty = TensorDataset(torch.zeros(0, 3, 224, 224), torch.zeros(0, dtype=torch.long))
        with pytest.raises(ValueError, match="empty"):
            save_embeddings(empty, m, tmp_path / "emb.pt")

    def test_creates_parent_dirs(self, tiny_dataset: TensorDataset, tmp_path: Path) -> None:
        """Output file parent directories must be created automatically."""
        m = BackboneEmbedder(pretrained=False).eval()
        out = tmp_path / "nested" / "dir" / "emb.pt"
        save_embeddings(tiny_dataset, m, out, batch_size=5)
        assert out.exists()
