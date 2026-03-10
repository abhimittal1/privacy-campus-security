"""
evaluate_lfw.py -- LFW benchmark evaluation for few-shot face recognition.

Two complementary evaluation modes:

1. **Pairs verification** (``evaluate_pairs``)
   Classic LFW protocol: 6,000 pairs -> cosine similarity -> binary match.
   Metrics: Accuracy, F1, ROC-AUC, TAR@FAR=0.1%, TAR@FAR=1%.
   Outputs: JSON metrics + ROC curve PNG.

2. **Few-shot (N-way K-shot)** (``evaluate_few_shot``)
   Episodic evaluation using FewShotEpisodicSampler.
   Metrics: Mean episode accuracy +/- 95% CI for K = {1, 2, 3, 5}.
   Outputs: JSON metrics per K + summary CSV.

Precomputed Embeddings
----------------------
Pass ``--embeddings path/to/embeddings.pt`` (saved by save_embeddings) to skip
backbone inference entirely.

Model Checkpoint
----------------
Pass ``--model-checkpoint path/to/checkpoint.pt`` (saved by ProtoNetTrainer)
to load fine-tuned backbone weights before evaluation.

CLI Examples
------------
::

    python -m src.eval.evaluate_lfw --mode pairs --root data/lfw
    python -m src.eval.evaluate_lfw --mode few-shot --n-way 5 --k-shot 5 --episodes 1000
    python -m src.eval.evaluate_lfw --mode both --embeddings experiments/emb.pt
    python -m src.eval.evaluate_lfw --mode few-shot \\
        --model-checkpoint experiments/checkpoints/best_checkpoint.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import LFWPairs

from src.data.lfw_loader import (
    DEFAULT_TRANSFORM,
    Episode,
    FewShotEpisodicSampler,
    LFWDataset,
    collate_episode,
)
from src.models.backbone import BackboneEmbedder

log = logging.getLogger(__name__)

_DEFAULT_OUTPUT_DIR = Path("experiments/sample_runs")


# ---------------------------------------------------------------------------
# Precomputed-embedding dataset shim
# ---------------------------------------------------------------------------


class _EmbeddingDataset(Dataset):
    """
    In-memory dataset wrapping pre-computed embedding tensors.

    Mimics the LFWDataset interface expected by FewShotEpisodicSampler:
    exposes .targets, .classes, and .num_classes.

    When used as the dataset in an episodic DataLoader, __getitem__ returns
    (embedding_1D, label_int, class_name_str).  The collate_episode function
    stacks these into Episode.support / Episode.query of shape (N*K, D), which
    can be passed directly through _IdentityNormalizer.

    Args:
        embeddings: (N, D) float tensor of L2-normalised embeddings.
        labels:     (N,) integer class-index tensor.
        classes:    Ordered list of class name strings.  Auto-generated when None.
    """

    def __init__(
        self,
        embeddings: Tensor,
        labels: Tensor,
        classes: Optional[list[str]] = None,
    ) -> None:
        self._embeddings = embeddings
        self._labels = labels.long()
        n_cls = int(self._labels.max().item()) + 1
        self.classes = classes if classes else [f"class_{i}" for i in range(n_cls)]
        self.targets = self._labels.tolist()
        self.num_classes = n_cls

    def __len__(self) -> int:
        return len(self._embeddings)

    def __getitem__(self, idx: int):  # noqa: ANN001
        return self._embeddings[idx], self._labels[idx], self.classes[self._labels[idx]]


class _IdentityNormalizer(nn.Module):
    """Passthrough backbone: L2-normalises an already-embedded 2-D input."""

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return F.normalize(x, p=2, dim=1)


# ---------------------------------------------------------------------------
# Helpers: TAR@FAR, metrics I/O
# ---------------------------------------------------------------------------


def _tar_at_far(
    similarities: np.ndarray, labels: np.ndarray, far_target: float
) -> float:
    """
    Compute True-Accept-Rate at a given False-Accept-Rate operating point.

    Args:
        similarities: Cosine similarity scores in [-1, 1].
        labels:       Ground-truth binary labels (1 = same person).
        far_target:   Target FAR, e.g. 0.001 for FAR=0.1%.

    Returns:
        Highest TAR achievable at FAR <= far_target.
    """
    impostor_sims = similarities[labels == 0]
    genuine_sims  = similarities[labels == 1]
    thresholds    = np.sort(similarities)[::-1]
    best_tar      = 0.0
    for t in thresholds:
        far = float((impostor_sims >= t).mean())
        if far <= far_target:
            tar      = float((genuine_sims >= t).mean())
            best_tar = max(best_tar, tar)
    return best_tar


def _log_metrics(metrics: dict[str, float], label: str) -> None:
    log.info("-- %s --", label)
    for k, v in metrics.items():
        log.info("  %-32s %.4f", k, v)


def _save_json(metrics: dict, filename: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    out_path.write_text(json.dumps(metrics, indent=2))
    log.info("Metrics -> %s", out_path)


def _save_csv(rows: list[dict], filename: str, output_dir: Path) -> None:
    """Write a list of metric dicts as a CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("CSV -> %s", out_path)


def _plot_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    output_dir: Path,
    filename: str = "roc_curve.png",
) -> None:
    """Save ROC curve PNG; silently skips when matplotlib is unavailable."""
    try:
        import matplotlib.pyplot as plt  # type: ignore[import]
    except ImportError:
        log.warning("matplotlib not installed -- ROC plot skipped.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Accept Rate")
    ax.set_ylabel("True Accept Rate")
    ax.set_title("LFW Pairs Verification -- ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out_path = output_dir / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("ROC plot -> %s", out_path)


# ---------------------------------------------------------------------------
# Internal builder helpers
# ---------------------------------------------------------------------------


def _load_model(model_checkpoint: Optional[str], device: str) -> BackboneEmbedder:
    """Construct BackboneEmbedder; optionally restore from a ProtoNetTrainer checkpoint."""
    model = BackboneEmbedder().to(device).eval()
    if model_checkpoint:
        ckpt_path = Path(model_checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        log.info("Backbone weights loaded from %s", ckpt_path)
    return model


def _load_embedding_dataset(embeddings_path: str) -> _EmbeddingDataset:
    """Load a save_embeddings() .pt file and wrap it in _EmbeddingDataset."""
    payload    = torch.load(embeddings_path, map_location="cpu")
    embeddings = payload["embeddings"]
    labels     = payload["labels"]
    classes    = payload.get("classes", [])
    log.info(
        "Loaded precomputed embeddings: %s | %d samples | dim=%d",
        embeddings_path, len(embeddings), embeddings.shape[1],
    )
    return _EmbeddingDataset(embeddings, labels, classes)


# ---------------------------------------------------------------------------
# Pairs verification evaluation
# ---------------------------------------------------------------------------


def evaluate_pairs(
    root: str | Path = "data/lfw",
    batch_size: int = 64,
    device: Optional[str] = None,
    model_checkpoint: Optional[str] = None,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    save_json: bool = True,
) -> dict[str, float]:
    """
    Evaluate backbone on the standard LFW pairs verification benchmark.

    Uses the 6,000 test pairs from torchvision's LFWPairs dataset.
    Cosine similarity between L2-normalised embeddings is the matching score.

    Args:
        root:             LFW data root directory (must contain downloaded data).
        batch_size:       Inference batch size.
        device:           "cuda" / "cpu"; auto-detected if None.
        model_checkpoint: Optional ProtoNetTrainer .pt path to restore weights.
        output_dir:       Directory for JSON and ROC plot.
        save_json:        Write metrics JSON + ROC PNG to output_dir.

    Returns:
        Dict: accuracy, f1, roc_auc, tar_at_far_0.1pct, tar_at_far_1pct.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Pairs evaluation | device=%s | checkpoint=%s", device, model_checkpoint)

    pairs_ds = LFWPairs(
        root=str(root),
        split="test",
        image_set="funneled",
        transform=DEFAULT_TRANSFORM,
        download=False,
    )
    loader = DataLoader(pairs_ds, batch_size=batch_size, num_workers=0, pin_memory=False)
    model  = _load_model(model_checkpoint, device)
    log.info("Evaluating %d pairs ...", len(pairs_ds))

    all_sims:   list[float] = []
    all_labels: list[int]   = []

    with torch.no_grad():
        for img1, img2, batch_labels in loader:
            e1   = model(img1.to(device))
            e2   = model(img2.to(device))
            sims = (e1 * e2).sum(dim=1).cpu().numpy()
            all_sims.extend(sims.tolist())
            all_labels.extend(batch_labels.numpy().tolist())

    sims_arr   = np.array(all_sims,   dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int32)
    preds      = (sims_arr >= 0.0).astype(int)
    fpr, tpr, _ = roc_curve(labels_arr, sims_arr)

    metrics: dict[str, float] = {
        "accuracy":          float(accuracy_score(labels_arr, preds)),
        "f1":                float(f1_score(labels_arr, preds, zero_division=0)),
        "roc_auc":           float(roc_auc_score(labels_arr, sims_arr)),
        "tar_at_far_0.1pct": _tar_at_far(sims_arr, labels_arr, far_target=0.001),
        "tar_at_far_1pct":   _tar_at_far(sims_arr, labels_arr, far_target=0.01),
    }
    _log_metrics(metrics, "Pairs Verification")
    if save_json:
        _save_json(metrics, "lfw_pairs_metrics.json", output_dir)
        _plot_roc(fpr, tpr, metrics["roc_auc"], output_dir)
    return metrics


# ---------------------------------------------------------------------------
# Few-shot evaluation
# ---------------------------------------------------------------------------


def evaluate_few_shot(
    root: str | Path = "data/lfw",
    n_way: int = 5,
    k_shot: int = 5,
    n_query: int = 15,
    num_episodes: int = 1000,
    device: Optional[str] = None,
    seed: int = 42,
    model_checkpoint: Optional[str] = None,
    embeddings_path: Optional[str] = None,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    save_json: bool = True,
) -> dict[str, float]:
    """
    N-way K-shot episodic evaluation on LFW test classes.

    Supports two modes:

    * **Raw images** -- backbone runs on every support/query image each episode.
    * **Precomputed** -- embeddings_path points to a save_embeddings() .pt file;
      backbone inference is replaced by L2 normalisation of precomputed vectors.

    Args:
        root:             LFW data root (unused when embeddings_path is set).
        n_way:            Classes per episode.
        k_shot:           Support images per class.
        n_query:          Query images per class.
        num_episodes:     Evaluation episodes.
        device:           Target device (auto if None).
        seed:             RNG seed for reproducible sampling.
        model_checkpoint: Optional ProtoNetTrainer .pt to restore backbone weights.
        embeddings_path:  Optional precomputed embeddings .pt file.
        output_dir:       Directory for JSON output.
        save_json:        Write metrics JSON to output_dir.

    Returns:
        Dict: mean_accuracy, ci95, n_way, k_shot, num_episodes.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        "Few-shot eval | %d-way %d-shot | %d eps | device=%s | emb=%s",
        n_way, k_shot, num_episodes, device, embeddings_path,
    )

    if embeddings_path:
        dataset: Dataset = _load_embedding_dataset(embeddings_path)
        model: nn.Module = _IdentityNormalizer().to(device)
    else:
        dataset = LFWDataset(
            root=root,
            split="test",
            download=False,
            min_images_per_class=k_shot + n_query,
        )
        model = _load_model(model_checkpoint, device)

    sampler = FewShotEpisodicSampler(
        dataset=dataset,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        num_episodes=num_episodes,
        class_split="test",
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        collate_fn=collate_episode,
    )

    model.eval()
    episode_accs: list[float] = []

    with torch.no_grad():
        for ep in loader:
            support  = ep.support.to(device)
            query    = ep.query.to(device)
            s_labels = ep.support_labels.to(device)
            q_labels = ep.query_labels

            support_embs: Tensor = model(support)
            query_embs:   Tensor = model(query)

            D          = support_embs.size(1)
            prototypes = torch.zeros(n_way, D, device=device)
            for cls in range(n_way):
                mask = s_labels == cls
                if mask.any():
                    prototypes[cls] = support_embs[mask].mean(dim=0)
            prototypes = F.normalize(prototypes, p=2, dim=1)

            logits = query_embs @ prototypes.T
            preds  = logits.argmax(dim=1).cpu()
            acc    = (preds == q_labels).float().mean().item()
            episode_accs.append(acc)

    accs     = np.array(episode_accs)
    mean_acc = float(accs.mean())
    ci95     = float(1.96 * accs.std() / np.sqrt(len(accs))) if len(accs) > 1 else 0.0

    metrics: dict[str, float] = {
        "mean_accuracy": mean_acc,
        "ci95":          ci95,
        "n_way":         float(n_way),
        "k_shot":        float(k_shot),
        "num_episodes":  float(num_episodes),
    }
    _log_metrics(metrics, f"{n_way}-way {k_shot}-shot Few-shot")
    if save_json:
        _save_json(metrics, f"lfw_few_shot_{n_way}w{k_shot}s_metrics.json", output_dir)
    return metrics


# ---------------------------------------------------------------------------
# K-sweep helper
# ---------------------------------------------------------------------------


def evaluate_k_sweep(
    k_values: list[int],
    n_way: int,
    num_episodes: int,
    output_dir: Path,
    **kwargs,
) -> list[dict]:
    """
    Run evaluate_few_shot for multiple K values and save a summary CSV.

    Args:
        k_values:     List of K (shots) to evaluate.
        n_way:        Fixed N for all runs.
        num_episodes: Episodes per K.
        output_dir:   Directory for CSV and per-K JSON files.
        **kwargs:     Extra keyword arguments forwarded to evaluate_few_shot.

    Returns:
        List of metric dicts, one per K value.
    """
    rows: list[dict] = []
    for k in k_values:
        m = evaluate_few_shot(
            n_way=n_way,
            k_shot=k,
            num_episodes=num_episodes,
            output_dir=output_dir,
            **kwargs,
        )
        rows.append({
            "k_shot":        k,
            "n_way":         n_way,
            "mean_accuracy": round(m["mean_accuracy"], 4),
            "ci95":          round(m["ci95"], 4),
            "num_episodes":  num_episodes,
        })
    _save_csv(rows, "lfw_few_shot_sweep.csv", output_dir)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LFW evaluation: pairs verification and/or N-way K-shot few-shot."
    )
    p.add_argument("--mode",             default="pairs",
                   choices=["pairs", "few-shot", "both"],
                   help="Evaluation mode (default: pairs)")
    p.add_argument("--root",             default="data/lfw",
                   help="LFW data root directory")
    p.add_argument("--batch-size",       type=int, default=64,
                   help="Batch size (pairs mode)")
    p.add_argument("--n-way",            type=int, default=5,
                   help="Classes per episode")
    p.add_argument("--k-shot",           type=int, default=5,
                   help="Shots per class (ignored for --mode both)")
    p.add_argument("--n-query",          type=int, default=15,
                   help="Queries per class")
    p.add_argument("--episodes",         type=int, default=1000,
                   help="Evaluation episodes per K (default: 1000)")
    p.add_argument("--device",           default=None,
                   help="cpu / cuda (auto-detect if omitted)")
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--embeddings",       default=None,
                   help="Path to precomputed embeddings .pt (from save_embeddings)")
    p.add_argument("--model-checkpoint", default=None,
                   help="Path to ProtoNetTrainer checkpoint .pt")
    p.add_argument("--output-dir",       default=str(_DEFAULT_OUTPUT_DIR),
                   help="Output directory for JSON, CSV, and ROC plot")
    p.add_argument("--no-save",          action="store_true",
                   help="Skip all file output")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args    = _parse_args()
    save    = not args.no_save
    out_dir = Path(args.output_dir)
    few_kw  = dict(
        root=args.root,
        n_way=args.n_way,
        n_query=args.n_query,
        num_episodes=args.episodes,
        device=args.device,
        seed=args.seed,
        model_checkpoint=args.model_checkpoint,
        embeddings_path=args.embeddings,
        output_dir=out_dir,
        save_json=save,
    )

    if args.mode in ("pairs", "both"):
        evaluate_pairs(
            root=args.root,
            batch_size=args.batch_size,
            device=args.device,
            model_checkpoint=args.model_checkpoint,
            output_dir=out_dir,
            save_json=save,
        )

    if args.mode == "few-shot":
        evaluate_few_shot(k_shot=args.k_shot, **few_kw)

    if args.mode == "both":
        evaluate_k_sweep(
            k_values=[1, 2, 3, 5],
            n_way=args.n_way,
            num_episodes=args.episodes,
            output_dir=out_dir,
            root=args.root,
            n_query=args.n_query,
            device=args.device,
            seed=args.seed,
            model_checkpoint=args.model_checkpoint,
            embeddings_path=args.embeddings,
            save_json=save,
        )


if __name__ == "__main__":
    main()
