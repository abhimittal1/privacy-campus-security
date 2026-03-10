"""
prototypical.py – Prototypical Network trainer and evaluator for few-shot face recognition.

Training uses episodic DataLoaders whose collate function yields
:class:`~src.data.lfw_loader.Episode` namedtuples.  Each episode is an
N-way K-shot Q-query task drawn from the meta-training class pool.

Key design decisions
--------------------
* **Cosine similarity** is the prototype-to-query metric because all backbone
  embeddings are L2-normalised.  For unit vectors this is equivalent to the
  negative squared Euclidean distance up to a constant, matching the spirit of
  Snell et al. (NeurIPS 2017).
* **Only the ProjectionHead** is updated by default
  (``ProtoNetConfig.freeze_backbone=True``).  Set ``False`` to fine-tune
  the full ResNet-50 body as well.
* **Logging** writes to both the console (INFO level) and an optional log file.

Reference
---------
Snell et al., "Prototypical Networks for Few-shot Learning", NeurIPS 2017.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from src.data.lfw_loader import Episode
from src.models.backbone import BackboneEmbedder

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ProtoNetConfig:
    """
    Hyper-parameters and runtime settings for :class:`ProtoNetTrainer`.

    Args:
        n_way:           Number of classes per episode.
        k_shot:          Support images per class.
        q_query:         Query images per class.
        learning_rate:   Optimizer learning rate.
        optimizer:       ``"adam"`` or ``"sgd"``.
        weight_decay:    L2 regularisation coefficient.
        epochs:          Total training epochs.
        device:          PyTorch device string (e.g. ``"cpu"`` or ``"cuda"``).
        freeze_backbone: If ``True``, only the projection head is updated;
                         the ResNet-50 body weights are frozen.
        log_file:        Optional filesystem path to write structured log output
                         in addition to console stderr.
        log_interval:    Print / log a running summary every *n* episodes processed
                         within a single epoch.
        save_best:       Automatically persist the best-validation checkpoint
                         during :meth:`ProtoNetTrainer.train`.
    """

    n_way: int = 5
    k_shot: int = 5
    q_query: int = 15
    learning_rate: float = 1e-3
    optimizer: str = "adam"   # "adam" | "sgd"
    weight_decay: float = 1e-4
    epochs: int = 50
    device: str = "cpu"
    freeze_backbone: bool = True
    log_file: Optional[str] = None
    log_interval: int = 10
    save_best: bool = True

    def __post_init__(self) -> None:
        if self.optimizer not in ("adam", "sgd"):
            raise ValueError(f"optimizer must be 'adam' or 'sgd', got '{self.optimizer}'")
        if self.n_way < 2:
            raise ValueError(f"n_way must be ≥ 2, got {self.n_way}")
        if self.k_shot < 1:
            raise ValueError(f"k_shot must be ≥ 1, got {self.k_shot}")
        if self.q_query < 1:
            raise ValueError(f"q_query must be ≥ 1, got {self.q_query}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class ProtoNetTrainer:
    """
    Trains and evaluates a Prototypical Network via episodic meta-learning.

    Args:
        backbone:     A :class:`~src.models.backbone.BackboneEmbedder` (or any
                      ``nn.Module`` mapping ``(B, C, H, W) → (B, D)`` L2-normalised).
        config:       :class:`ProtoNetConfig` with all training hyper-parameters.
        train_loader: DataLoader whose ``collate_fn`` is
                      :func:`~src.data.lfw_loader.collate_episode`; yields
                      :class:`~src.data.lfw_loader.Episode` namedtuples.
        val_loader:   Optional validation DataLoader (same format as *train_loader*).

    Example::

        config  = ProtoNetConfig(n_way=5, k_shot=1, q_query=10, epochs=20)
        trainer = ProtoNetTrainer(backbone, config, train_loader, val_loader)
        history = trainer.train(save_dir="experiments/checkpoints")
    """

    def __init__(
        self,
        backbone: BackboneEmbedder,
        config: ProtoNetConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device(config.device)
        self.net = backbone.to(self.device)

        # Freeze backbone body; projection head is always trainable.
        if config.freeze_backbone:
            for param in self.net.body.parameters():
                param.requires_grad = False
            trainable_params = list(self.net.projector.parameters())
        else:
            trainable_params = list(self.net.parameters())

        if config.optimizer == "adam":
            self.optimizer: torch.optim.Optimizer = Adam(
                trainable_params,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = SGD(
                trainable_params,
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay,
            )

        self._best_val_acc: float = -1.0
        self._setup_logging()

    # ------------------------------------------------------------------
    # Logging setup
    # ------------------------------------------------------------------

    def _setup_logging(self) -> None:
        """Configure console handler and optional file handler."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(ch)
        if self.config.log_file:
            log_path = Path(self.config.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(fh)
            logger.info("Logging to file: %s", log_path)

    # ------------------------------------------------------------------
    # Core prototypical computation
    # ------------------------------------------------------------------

    def _episode_loss_and_acc(self, episode: Episode) -> tuple[Tensor, float]:
        """
        Compute cross-entropy loss and episode accuracy for a single episode.

        Algorithm:

        1. Embed all support images via the backbone → ``(N*K, D)``.
        2. Per class average the support embeddings → prototype, L2-renormalise.
        3. Embed query images → ``(N*Q, D)``.
        4. Cosine similarity: ``logits = query_embs @ prototypes.T`` → ``(N*Q, N)``.
        5. ``loss = cross_entropy(logits, local_query_labels)``.
        6. ``acc  = mean(argmax(logits) == local_query_labels)``.

        Args:
            episode: An :class:`~src.data.lfw_loader.Episode` namedtuple.

        Returns:
            ``(loss, accuracy)`` – *loss* is a differentiable Tensor;
            *accuracy* is a Python float.
        """
        support  = episode.support.to(self.device)        # (N*K, C, H, W)
        query    = episode.query.to(self.device)          # (N*Q, C, H, W)
        s_labels = episode.support_labels.to(self.device) # (N*K,) local 0…N-1
        q_labels = episode.query_labels.to(self.device)   # (N*Q,) local 0…N-1

        n_way = self.config.n_way

        support_embs: Tensor = self.net(support)   # (N*K, D)
        query_embs:   Tensor = self.net(query)     # (N*Q, D)

        # Build per-class prototypes (mean → re-normalise).
        D = support_embs.size(1)
        prototypes = torch.zeros(n_way, D, device=self.device)
        for cls in range(n_way):
            mask = s_labels == cls
            if mask.any():
                prototypes[cls] = support_embs[mask].mean(dim=0)
        prototypes = F.normalize(prototypes, p=2, dim=1)  # (N, D)

        # Cosine similarity logits and cross-entropy loss.
        logits: Tensor = query_embs @ prototypes.T  # (N*Q, N)
        loss:   Tensor = F.cross_entropy(logits, q_labels)

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc   = (preds == q_labels).float().mean().item()

        return loss, acc

    # ------------------------------------------------------------------
    # Public training API
    # ------------------------------------------------------------------

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """
        Run one full pass over *train_loader* in episodic mode.

        The backbone body is kept in eval mode when ``config.freeze_backbone``
        is ``True`` to prevent BatchNorm running-stats from drifting.

        Args:
            epoch: Current epoch number (1-based; used only for log messages).

        Returns:
            Dictionary with keys ``"loss"``, ``"accuracy"``, ``"episodes"``,
            and ``"elapsed_s"`` (epoch averages / totals).
        """
        self.net.train()
        if self.config.freeze_backbone:
            # Keep BN statistics stable while body weights are frozen.
            self.net.body.eval()

        total_loss = 0.0
        total_acc  = 0.0
        ep_idx     = 0
        t0         = time.perf_counter()

        for ep_idx, episode in enumerate(self.train_loader, start=1):
            self.optimizer.zero_grad()
            loss, acc = self._episode_loss_and_acc(episode)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_acc  += acc

            if ep_idx % self.config.log_interval == 0:
                log.info(
                    "Epoch %d | episode %4d | loss %.4f | acc %.3f",
                    epoch, ep_idx, loss.item(), acc,
                )

        n_ep    = max(ep_idx, 1)
        elapsed = time.perf_counter() - t0
        metrics = {
            "loss":      total_loss / n_ep,
            "accuracy":  total_acc  / n_ep,
            "episodes":  float(n_ep),
            "elapsed_s": elapsed,
        }
        log.info(
            "Epoch %d TRAIN | loss %.4f | acc %.3f | %d eps | %.1fs",
            epoch, metrics["loss"], metrics["accuracy"], n_ep, elapsed,
        )
        return metrics

    @torch.no_grad()
    def validate_episode(self, episode: Episode) -> float:
        """
        Evaluate accuracy on a single episode without gradient computation.

        Args:
            episode: An :class:`~src.data.lfw_loader.Episode` namedtuple.

        Returns:
            Episode accuracy as a float in ``[0, 1]``.
        """
        self.net.eval()
        _, acc = self._episode_loss_and_acc(episode)
        return acc

    def validate(self) -> dict[str, float]:
        """
        Run the full *val_loader* and return averaged accuracy ± 95% CI.

        Returns an empty dict if *val_loader* was not supplied.

        Returns:
            Dictionary with ``"mean_accuracy"``, ``"ci95"``, and
            ``"n_episodes"``.
        """
        if self.val_loader is None:
            return {}

        self.net.eval()
        accs: list[float] = []
        with torch.no_grad():
            for episode in self.val_loader:
                _, acc = self._episode_loss_and_acc(episode)
                accs.append(acc)

        arr      = np.array(accs)
        mean_acc = float(arr.mean())
        ci95     = float(1.96 * arr.std() / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        log.info("VAL  | acc %.3f ± %.3f (n=%d episodes)", mean_acc, ci95, len(arr))
        return {"mean_accuracy": mean_acc, "ci95": ci95, "n_episodes": float(len(arr))}

    def train(self, save_dir: str | Path = "experiments/checkpoints") -> list[dict]:
        """
        Full training loop across ``config.epochs`` epochs.

        Each epoch:

        1. :meth:`train_one_epoch` — SGD/Adam over the training DataLoader.
        2. :meth:`validate`        — accuracy on the validation DataLoader (if given).
        3. If ``config.save_best`` and the new val accuracy is a new best, save a
           ``best_checkpoint.pt`` under *save_dir*.

        A ``final_checkpoint.pt`` is always saved at the end.

        Args:
            save_dir: Directory to persist checkpoints.

        Returns:
            List of per-epoch metric dicts (keys are the union of
            :meth:`train_one_epoch` and :meth:`validate` output keys,
            plus ``"epoch"``).
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict] = []

        log.info(
            "Starting training | %d-way %d-shot | %d epochs | device=%s",
            self.config.n_way, self.config.k_shot, self.config.epochs, self.config.device,
        )

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self.train_one_epoch(epoch)
            val_metrics   = self.validate()
            epoch_metrics = {"epoch": epoch, **train_metrics, **val_metrics}
            history.append(epoch_metrics)

            if self.config.save_best and val_metrics:
                val_acc = val_metrics.get("mean_accuracy", 0.0)
                if val_acc > self._best_val_acc:
                    self._best_val_acc = val_acc
                    best_path = save_dir / "best_checkpoint.pt"
                    self.save_checkpoint(best_path, epoch=epoch, metrics=epoch_metrics)
                    log.info("New best val acc %.4f → saved %s", val_acc, best_path)

        final_path = save_dir / "final_checkpoint.pt"
        self.save_checkpoint(final_path, epoch=self.config.epochs, metrics=history[-1])
        log.info("Training complete. Final checkpoint: %s", final_path)
        return history

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(
        self, path: str | Path, epoch: int, metrics: dict
    ) -> None:
        """
        Persist model and optimizer states to disk.

        The saved file is a ``dict`` with keys:
        ``epoch``, ``metrics``, ``config``, ``model_state_dict``,
        ``optimizer_state_dict``.

        Args:
            path:    Destination ``.pt`` file path (parent dirs are created).
            epoch:   Current epoch number (stored as metadata).
            metrics: Arbitrary metric dict to embed in the checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch":                epoch,
            "metrics":              metrics,
            "config":               dataclasses.asdict(self.config),
            "model_state_dict":     self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(payload, path)
        log.info("Checkpoint saved → %s (epoch %d)", path, epoch)

    def load_checkpoint(self, path: str | Path) -> dict:
        """
        Restore model and optimizer state from a ``.pt`` checkpoint.

        Args:
            path: Path to a file previously created by :meth:`save_checkpoint`.

        Returns:
            The full checkpoint ``dict`` (``epoch``, ``metrics``,
            ``config``, …).

        Raises:
            FileNotFoundError: If *path* does not point to an existing file.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        payload = torch.load(path, map_location=self.device)
        self.net.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        log.info(
            "Checkpoint loaded ← %s (epoch %s)", path, payload.get("epoch", "?")
        )
        return payload

