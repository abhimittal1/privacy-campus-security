"""
test_prototypical.py -- Unit tests for ProtoNetConfig, ProtoNetTrainer,
and the evaluate_lfw helpers (_EmbeddingDataset, _IdentityNormalizer).

All tests run on CPU with randomly-initialised (pretrained=False) backbones,
so no network access or dataset downloads are required.

Synthetic episode strategy
--------------------------
To produce deterministic, non-trivial accuracy without training:

* For each class c in {0 ... N-1} we construct a "canonical" 512-D unit vector
  e_c (orthonormal basis vector in a random direction, seeded).
* Support images are tiny (3, 16, 16) tensors whose *mean pixel value* encodes
  which canonical embedding the backbone should reproduce -- in practice, with
  random weights the backbone output won't match e_c, BUT for the
  ``test_nontrivial_accuracy`` test we mock the backbone's forward pass to
  return the canonical embeddings directly, guaranteeing acc == 1.0.

Tests that only check shapes/types use real (random) backbone outputs.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from src.data.lfw_loader import Episode, collate_episode
from src.eval.evaluate_lfw import _EmbeddingDataset, _IdentityNormalizer
from src.models.backbone import BackboneEmbedder, EMBEDDING_DIM
from src.models.prototypical import ProtoNetConfig, ProtoNetTrainer


# ---------------------------------------------------------------------------
# Synthetic episode helpers
# ---------------------------------------------------------------------------

N_WAY   = 3
K_SHOT  = 2
Q_QUERY = 4
IMG_C   = 3
IMG_H   = 16
IMG_W   = 16


def _make_episode(n_way: int = N_WAY, k_shot: int = K_SHOT, q_query: int = Q_QUERY) -> Episode:
    """Return a synthetic Episode with random image tensors and sequential labels."""
    n_support   = n_way * k_shot
    n_query_tot = n_way * q_query

    support = torch.randn(n_support, IMG_C, IMG_H, IMG_W)
    query   = torch.randn(n_query_tot, IMG_C, IMG_H, IMG_W)

    # Local labels: 0,0,...,1,1,...,N-1,N-1,...
    s_labels = torch.repeat_interleave(torch.arange(n_way), k_shot)
    q_labels = torch.repeat_interleave(torch.arange(n_way), q_query)

    class_names = [f"person_{i}" for i in range(n_way)]
    return Episode(support, query, s_labels, q_labels, class_names)


def _make_loader(
    n_episodes: int = 4,
    n_way: int = N_WAY,
    k_shot: int = K_SHOT,
    q_query: int = Q_QUERY,
) -> DataLoader:
    """Return a DataLoader that yields *n_episodes* pre-built synthetic Episodes."""
    episodes = [_make_episode(n_way, k_shot, q_query) for _ in range(n_episodes)]
    # Wrap in a plain list-based DataLoader using a passthrough collate.
    return DataLoader(episodes, batch_size=1, collate_fn=lambda b: b[0])


def _make_config(**overrides) -> ProtoNetConfig:
    defaults = dict(
        n_way=N_WAY,
        k_shot=K_SHOT,
        q_query=Q_QUERY,
        learning_rate=1e-3,
        optimizer="adam",
        epochs=1,
        device="cpu",
        freeze_backbone=True,
        log_interval=100,  # suppress mid-epoch logs in tests
        save_best=False,
    )
    defaults.update(overrides)
    return ProtoNetConfig(**defaults)


def _make_trainer(
    train_loader: DataLoader | None = None,
    val_loader: DataLoader | None = None,
    **config_overrides,
) -> ProtoNetTrainer:
    backbone = BackboneEmbedder(pretrained=False)
    cfg      = _make_config(**config_overrides)
    tl       = train_loader if train_loader is not None else _make_loader()
    return ProtoNetTrainer(backbone, cfg, tl, val_loader)


# ---------------------------------------------------------------------------
# ProtoNetConfig
# ---------------------------------------------------------------------------


class TestProtoNetConfig:
    def test_defaults_valid(self) -> None:
        cfg = ProtoNetConfig()
        assert cfg.n_way == 5
        assert cfg.k_shot == 5
        assert cfg.q_query == 15
        assert cfg.optimizer in ("adam", "sgd")

    def test_custom_values(self) -> None:
        cfg = ProtoNetConfig(n_way=3, k_shot=1, q_query=5, epochs=10, device="cpu")
        assert cfg.n_way == 3
        assert cfg.k_shot == 1
        assert cfg.q_query == 5
        assert cfg.epochs == 10

    def test_invalid_optimizer_raises(self) -> None:
        with pytest.raises(ValueError, match="optimizer"):
            ProtoNetConfig(optimizer="rmsprop")  # type: ignore[arg-type]

    def test_n_way_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="n_way"):
            ProtoNetConfig(n_way=1)

    def test_k_shot_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="k_shot"):
            ProtoNetConfig(k_shot=0)

    def test_q_query_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="q_query"):
            ProtoNetConfig(q_query=0)

    def test_dataclass_asdict(self) -> None:
        """dataclasses.asdict must work (needed for checkpoint serialisation)."""
        cfg = _make_config()
        d   = dataclasses.asdict(cfg)
        assert "n_way" in d
        assert "learning_rate" in d

    def test_sgd_optimizer_accepted(self) -> None:
        cfg = ProtoNetConfig(optimizer="sgd")
        assert cfg.optimizer == "sgd"


# ---------------------------------------------------------------------------
# ProtoNetTrainer -- construction and attribute checks
# ---------------------------------------------------------------------------


class TestProtoNetTrainerInit:
    def test_device_attribute(self) -> None:
        trainer = _make_trainer()
        assert str(trainer.device) == "cpu"

    def test_freeze_backbone_true(self) -> None:
        trainer = _make_trainer(freeze_backbone=True)
        for p in trainer.net.body.parameters():
            assert not p.requires_grad, "body must be frozen"

    def test_freeze_backbone_false(self) -> None:
        trainer = _make_trainer(freeze_backbone=False)
        has_grad = any(p.requires_grad for p in trainer.net.body.parameters())
        assert has_grad, "at least some body params should require grad"

    def test_projector_always_trainable(self) -> None:
        for freeze in (True, False):
            trainer = _make_trainer(freeze_backbone=freeze)
            for p in trainer.net.projector.parameters():
                assert p.requires_grad, "projector must always be trainable"

    def test_adam_optimizer_created(self) -> None:
        trainer = _make_trainer(optimizer="adam")
        assert isinstance(trainer.optimizer, torch.optim.Adam)

    def test_sgd_optimizer_created(self) -> None:
        trainer = _make_trainer(optimizer="sgd")
        assert isinstance(trainer.optimizer, torch.optim.SGD)

    def test_val_loader_none_by_default(self) -> None:
        trainer = _make_trainer()
        assert trainer.val_loader is None


# ---------------------------------------------------------------------------
# ProtoNetTrainer._episode_loss_and_acc
# ---------------------------------------------------------------------------


class TestEpisodeLossAndAcc:
    @pytest.fixture(scope="class")
    def trainer(self) -> ProtoNetTrainer:
        return _make_trainer()

    def test_loss_is_tensor(self, trainer: ProtoNetTrainer) -> None:
        ep   = _make_episode()
        loss, _ = trainer._episode_loss_and_acc(ep)
        assert isinstance(loss, torch.Tensor)

    def test_loss_is_finite(self, trainer: ProtoNetTrainer) -> None:
        ep   = _make_episode()
        loss, _ = trainer._episode_loss_and_acc(ep)
        assert torch.isfinite(loss)

    def test_loss_scalar(self, trainer: ProtoNetTrainer) -> None:
        ep   = _make_episode()
        loss, _ = trainer._episode_loss_and_acc(ep)
        assert loss.ndim == 0

    def test_acc_is_float(self, trainer: ProtoNetTrainer) -> None:
        ep   = _make_episode()
        _, acc = trainer._episode_loss_and_acc(ep)
        assert isinstance(acc, float)

    def test_acc_range(self, trainer: ProtoNetTrainer) -> None:
        ep   = _make_episode()
        _, acc = trainer._episode_loss_and_acc(ep)
        assert 0.0 <= acc <= 1.0

    def test_loss_differentiable(self, trainer: ProtoNetTrainer) -> None:
        """Loss must carry a grad_fn so backprop works."""
        trainer.net.train()
        ep   = _make_episode()
        loss, _ = trainer._episode_loss_and_acc(ep)
        assert loss.grad_fn is not None

    def test_nontrivial_accuracy(self) -> None:
        """
        When backbone returns known, perfectly-separable embeddings,
        accuracy must be 1.0.

        Strategy: mock BackboneEmbedder.forward to return a canonical per-class
        embedding derived from the support_labels/query_labels stored in the
        episode.  Because prototypical matching is exact in this case, acc == 1.
        """
        n_way   = 3
        k_shot  = 2
        q_query = 4
        D       = EMBEDDING_DIM

        # Build N_WAY orthogonal unit vectors.
        torch.manual_seed(0)
        basis = F.normalize(torch.eye(n_way, D), p=2, dim=1)  # (N, D)

        ep = _make_episode(n_way, k_shot, q_query)
        # Map each image's position to its class label.
        all_labels = torch.cat([ep.support_labels, ep.query_labels])
        n_total    = len(all_labels)

        call_count = [0]

        def _fake_forward(images: Tensor) -> Tensor:
            """Return the basis vector for each image's local label."""
            b = images.size(0)
            start = call_count[0]
            idxs  = all_labels[start: start + b]
            call_count[0] += b
            return basis[idxs]

        backbone = BackboneEmbedder(pretrained=False)
        cfg      = _make_config(n_way=n_way, k_shot=k_shot, q_query=q_query)
        trainer  = ProtoNetTrainer(backbone, cfg, _make_loader())
        trainer.net.forward = _fake_forward  # type: ignore[method-assign]

        _, acc = trainer._episode_loss_and_acc(ep)
        assert acc == pytest.approx(1.0), f"Expected acc=1.0, got {acc}"


# ---------------------------------------------------------------------------
# train_one_epoch
# ---------------------------------------------------------------------------


class TestTrainOneEpoch:
    @pytest.fixture(scope="class")
    def trained_metrics(self) -> dict:
        loader  = _make_loader(n_episodes=6)
        trainer = _make_trainer(train_loader=loader)
        return trainer.train_one_epoch(epoch=1)

    def test_returns_dict(self, trained_metrics: dict) -> None:
        assert isinstance(trained_metrics, dict)

    def test_has_loss_key(self, trained_metrics: dict) -> None:
        assert "loss" in trained_metrics

    def test_has_accuracy_key(self, trained_metrics: dict) -> None:
        assert "accuracy" in trained_metrics

    def test_has_episodes_key(self, trained_metrics: dict) -> None:
        assert "episodes" in trained_metrics

    def test_has_elapsed_key(self, trained_metrics: dict) -> None:
        assert "elapsed_s" in trained_metrics

    def test_loss_finite(self, trained_metrics: dict) -> None:
        assert torch.isfinite(torch.tensor(trained_metrics["loss"]))

    def test_accuracy_in_range(self, trained_metrics: dict) -> None:
        assert 0.0 <= trained_metrics["accuracy"] <= 1.0

    def test_episodes_equals_loader_length(self, trained_metrics: dict) -> None:
        assert trained_metrics["episodes"] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# validate_episode
# ---------------------------------------------------------------------------


class TestValidateEpisode:
    def test_returns_float(self) -> None:
        trainer = _make_trainer()
        ep      = _make_episode()
        acc     = trainer.validate_episode(ep)
        assert isinstance(acc, float)

    def test_in_range(self) -> None:
        trainer = _make_trainer()
        ep      = _make_episode()
        acc     = trainer.validate_episode(ep)
        assert 0.0 <= acc <= 1.0

    def test_no_gradient_side_effects(self) -> None:
        """validate_episode must not accumulate gradients on parameters."""
        trainer  = _make_trainer(freeze_backbone=False)
        # Zero-out gradients first.
        trainer.optimizer.zero_grad()
        ep = _make_episode()
        trainer.validate_episode(ep)
        for p in trainer.net.projector.parameters():
            assert p.grad is None or (p.grad == 0).all()


# ---------------------------------------------------------------------------
# validate (full val_loader)
# ---------------------------------------------------------------------------


class TestValidate:
    def test_empty_when_no_val_loader(self) -> None:
        trainer = _make_trainer(val_loader=None)
        result  = trainer.validate()
        assert result == {}

    def test_returns_mean_accuracy(self) -> None:
        val_loader = _make_loader(n_episodes=4)
        trainer    = _make_trainer(val_loader=val_loader)
        result     = trainer.validate()
        assert "mean_accuracy" in result
        assert 0.0 <= result["mean_accuracy"] <= 1.0

    def test_returns_ci95(self) -> None:
        val_loader = _make_loader(n_episodes=4)
        trainer    = _make_trainer(val_loader=val_loader)
        result     = trainer.validate()
        assert "ci95" in result
        assert result["ci95"] >= 0.0


# ---------------------------------------------------------------------------
# save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------


class TestCheckpointRoundtrip:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        trainer  = _make_trainer()
        out_path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(out_path, epoch=1, metrics={"loss": 1.5})
        assert out_path.exists()

    def test_load_restores_weights(self, tmp_path: Path) -> None:
        """After save + load the model state_dict must match."""
        trainer_a = _make_trainer()
        out_path  = tmp_path / "ckpt.pt"
        trainer_a.save_checkpoint(out_path, epoch=5, metrics={"accuracy": 0.75})

        trainer_b = _make_trainer()
        trainer_b.load_checkpoint(out_path)

        for (ka, va), (kb, vb) in zip(
            trainer_a.net.state_dict().items(),
            trainer_b.net.state_dict().items(),
        ):
            assert ka == kb
            assert torch.allclose(va, vb), f"Mismatch at key {ka}"

    def test_load_returns_payload(self, tmp_path: Path) -> None:
        trainer  = _make_trainer()
        out_path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(out_path, epoch=3, metrics={"loss": 0.8})
        payload  = trainer.load_checkpoint(out_path)
        assert payload["epoch"] == 3
        assert "model_state_dict" in payload
        assert "optimizer_state_dict" in payload
        assert "config" in payload

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        trainer = _make_trainer()
        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint(tmp_path / "nonexistent.pt")

    def test_config_serialised_in_payload(self, tmp_path: Path) -> None:
        trainer  = _make_trainer(n_way=4, k_shot=3)
        out_path = tmp_path / "ckpt.pt"
        trainer.save_checkpoint(out_path, epoch=1, metrics={})
        payload  = trainer.load_checkpoint(out_path)
        assert payload["config"]["n_way"] == 4
        assert payload["config"]["k_shot"] == 3

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        trainer  = _make_trainer()
        out_path = tmp_path / "nested" / "dir" / "ckpt.pt"
        trainer.save_checkpoint(out_path, epoch=1, metrics={})
        assert out_path.exists()


# ---------------------------------------------------------------------------
# evaluate_lfw helpers
# ---------------------------------------------------------------------------


class TestEmbeddingDataset:
    @pytest.fixture()
    def ds(self) -> _EmbeddingDataset:
        emb = F.normalize(torch.randn(20, EMBEDDING_DIM), p=2, dim=1)
        lbl = torch.arange(20) % 5
        return _EmbeddingDataset(emb, lbl)

    def test_len(self, ds: _EmbeddingDataset) -> None:
        assert len(ds) == 20

    def test_num_classes(self, ds: _EmbeddingDataset) -> None:
        assert ds.num_classes == 5

    def test_targets_length(self, ds: _EmbeddingDataset) -> None:
        assert len(ds.targets) == 20

    def test_getitem_returns_tuple(self, ds: _EmbeddingDataset) -> None:
        emb, lbl, name = ds[0]
        assert isinstance(emb, Tensor)
        assert isinstance(lbl, Tensor)
        assert isinstance(name, str)

    def test_embedding_shape(self, ds: _EmbeddingDataset) -> None:
        emb, _, _ = ds[0]
        assert emb.shape == (EMBEDDING_DIM,)

    def test_custom_classes(self) -> None:
        emb = torch.randn(6, 32)
        lbl = torch.tensor([0, 0, 1, 1, 2, 2])
        cls = ["alice", "bob", "charlie"]
        ds  = _EmbeddingDataset(emb, lbl, classes=cls)
        assert ds.classes == cls

    def test_auto_classes_generated(self, ds: _EmbeddingDataset) -> None:
        assert all(c.startswith("class_") for c in ds.classes)


class TestIdentityNormalizer:
    def test_output_shape(self) -> None:
        net = _IdentityNormalizer()
        x   = torch.randn(8, EMBEDDING_DIM)
        out = net(x)
        assert out.shape == (8, EMBEDDING_DIM)

    def test_l2_normalised(self) -> None:
        net  = _IdentityNormalizer()
        x    = torch.randn(5, EMBEDDING_DIM)
        out  = net(x)
        norms = out.norm(dim=1)
        assert torch.allclose(norms, torch.ones(5), atol=1e-5)
