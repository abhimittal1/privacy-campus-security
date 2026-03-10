"""
run.py – Unified CLI entry point for the Privacy Campus Security system.

Usage
-----
    # Pipeline subcommands (baseline orchestration)
    pcs prepare-data        --dataset lfw --out-dir data/lfw
    pcs compute-embeddings  [--checkpoint ckpt.pt] --out embeddings/lfw_embeddings.npz
    pcs train-protonet      --config configs/proto.yaml
    pcs eval-lfw            --embeddings embeddings/lfw_embeddings.npz \\
                            --episodes 1000 --k 1 2 3 5 \\
                            --out experiments/lfw_results

    # Operational subcommands
    pcs enroll  --subject-id <id> --image <path>
    pcs enroll-batch --image <path> [--group-id <gid>]
    pcs verify  --subject-id <id> --image <path>
    pcs eval    [--root data/lfw] [--batch-size 64]
    pcs serve   [--port 5000]

Each pipeline step is independently re-runnable: outputs from an earlier step
are consumed by later steps, but every function can also be called in isolation.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _setup_logging(log_file: Optional[str] = None) -> None:
    """Configure root logger with optional file handler."""
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout)
    ]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _auto_device() -> str:
    """Return 'cuda' when a GPU is available, else 'cpu'."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_embeddings_npz(npz_path: str | Path):
    """
    Load embeddings from a .npz file saved by ``cmd_compute_embeddings``.

    Returns
    -------
    embeddings : torch.Tensor  shape (N, D)
    labels     : torch.Tensor  shape (N,)
    classes    : list[str]
    """
    import numpy as np
    import torch

    data = np.load(str(npz_path), allow_pickle=True)
    embeddings = torch.tensor(data["embeddings"], dtype=torch.float32)
    labels = torch.tensor(data["labels"], dtype=torch.long)
    classes = data["classes"].tolist() if "classes" in data else []
    return embeddings, labels, classes


# ── Pipeline step 1 – prepare-data ───────────────────────────────────────────


def prepare_data(dataset: str, out_dir: str | Path) -> Path:
    """
    Download and verify the requested dataset.

    Currently supports ``lfw`` (Labeled Faces in the Wild).  Torchvision
    downloads the data once and skips re-downloading on subsequent calls.

    Args:
        dataset: Dataset name; only ``"lfw"`` is supported now.
        out_dir: Root directory where raw data will be stored.

    Returns:
        Resolved path to the downloaded dataset root.
    """
    out_dir = Path(out_dir).resolve()
    if dataset.lower() != "lfw":
        raise ValueError(f"Unsupported dataset '{dataset}'. Only 'lfw' is supported.")

    log.info("Preparing dataset '%s' → %s", dataset, out_dir)

    from src.data.lfw_loader import LFWDataset

    ds = LFWDataset(root=out_dir, split="train", download=True)
    log.info(
        "LFW ready: %d training images across %d identities.",
        len(ds),
        ds.num_classes,
    )
    return out_dir


def cmd_prepare_data(args: argparse.Namespace) -> None:
    prepare_data(dataset=args.dataset, out_dir=args.out_dir)


# ── Pipeline step 2 – compute-embeddings ─────────────────────────────────────


def compute_embeddings(
    root: str | Path = "data/lfw",
    checkpoint: Optional[str] = None,
    out: str | Path = "embeddings/lfw_embeddings.npz",
    batch_size: int = 64,
    device: Optional[str] = None,
) -> Path:
    """
    Run the backbone over every LFW image and save L2-normalised embeddings.

    The backbone is loaded from *checkpoint* when supplied; otherwise
    ImageNet-pretrained weights are used.  Embeddings are saved in NumPy
    ``.npz`` format (``embeddings``, ``labels``, ``classes`` arrays) so that
    they can be consumed by subsequent pipeline steps without PyTorch.

    Args:
        root:       LFW data root (must already be downloaded).
        checkpoint: Optional ProtoNetTrainer checkpoint ``.pt`` to restore weights.
        out:        Output ``.npz`` path.
        batch_size: Inference batch size.
        device:     PyTorch device string; auto-detected when ``None``.

    Returns:
        Resolved path to the saved ``.npz`` file.
    """
    import numpy as np

    from src.data.lfw_loader import LFWDataset
    from src.models.backbone import BackboneEmbedder
    from src.models.embedding_utils import save_embeddings

    device = device or _auto_device()
    out_path = Path(out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Computing embeddings | root=%s | checkpoint=%s | device=%s", root, checkpoint, device)

    # Load model ------------------------------------------------------------------
    model = BackboneEmbedder()
    if checkpoint:
        import torch
        ckpt = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        log.info("Loaded backbone weights from %s", checkpoint)
    model.eval()

    # Compute for both splits and concatenate ------------------------------------
    all_embeddings, all_labels, class_names = [], [], []
    for split in ("train", "test"):
        try:
            ds = LFWDataset(root=root, split=split, download=False)
        except Exception as exc:
            log.warning("Split '%s' skipped: %s", split, exc)
            continue

        pt_tmp = out_path.with_suffix(".pt")
        payload = save_embeddings(
            dataset=ds,
            model=model,
            out_path=pt_tmp,
            batch_size=batch_size,
            device=device,
        )

        import torch
        emb = payload["embeddings"]
        lbl = payload["labels"]
        # Offset labels to avoid collisions across splits
        offset = max(all_labels[-1].max().item() + 1, 0) if all_labels else 0
        all_embeddings.append(emb.numpy())
        all_labels.append(lbl.numpy() + offset)
        if not class_names:
            class_names = payload.get("classes", []) or []
        else:
            class_names += payload.get("classes", []) or []

    if not all_embeddings:
        raise RuntimeError("No embeddings computed – check your --root path.")

    import numpy as np

    embeddings_np = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    labels_np = np.concatenate(all_labels, axis=0).astype(np.int64)

    np.savez(
        str(out_path),
        embeddings=embeddings_np,
        labels=labels_np,
        classes=np.array(class_names, dtype=object),
    )
    log.info(
        "Saved %d embeddings (dim=%d) → %s",
        embeddings_np.shape[0],
        embeddings_np.shape[1],
        out_path,
    )
    return out_path


def cmd_compute_embeddings(args: argparse.Namespace) -> None:
    compute_embeddings(
        root=args.root,
        checkpoint=args.checkpoint,
        out=args.out,
        batch_size=args.batch_size,
        device=args.device,
    )


# ── Pipeline step 3 – train-protonet ─────────────────────────────────────────


def train_protonet(
    config_path: str | Path,
    data_root: str | Path = "data/lfw",
    out_dir: str | Path = "experiments/checkpoints",
    device: Optional[str] = None,
) -> Path:
    """
    Train a Prototypical Network using the hyperparameters in *config_path*.

    The YAML file is parsed into a :class:`~src.models.prototypical.ProtoNetConfig`
    and then a :class:`~src.models.prototypical.ProtoNetTrainer` runs the
    episodic training loop.  The best checkpoint is written to *out_dir*.

    Args:
        config_path: Path to a ``configs/proto.yaml`` file.
        data_root:   LFW data root directory.
        out_dir:     Directory for saved checkpoints and training logs.
        device:      PyTorch device override (overrides the value in YAML).

    Returns:
        Path to the best saved checkpoint (``.pt``).
    """
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        raise ImportError("PyYAML is required for train-protonet. Run: pip install pyyaml")

    import torch
    from torch.utils.data import DataLoader

    from src.data.lfw_loader import (
        FewShotEpisodicSampler,
        LFWDataset,
        collate_episode,
    )
    from src.models.backbone import BackboneEmbedder
    from src.models.prototypical import ProtoNetConfig, ProtoNetTrainer

    # Load YAML -------------------------------------------------------------------
    config_path = Path(config_path).resolve()
    log.info("Loading ProtoNet config from %s", config_path)
    with config_path.open() as fh:
        raw = yaml.safe_load(fh)

    cfg_dict = raw.get("protonet", raw)  # allow top-level or nested key
    if device:
        cfg_dict["device"] = device
    else:
        cfg_dict.setdefault("device", _auto_device())

    seed = raw.get("seed", 42)
    torch.manual_seed(seed)

    config = ProtoNetConfig(**{
        k: v for k, v in cfg_dict.items()
        if k in ProtoNetConfig.__dataclass_fields__
    })
    log.info("Config: %s", config)

    # Build loaders ---------------------------------------------------------------
    min_imgs = config.k_shot + config.q_query
    train_ds = LFWDataset(root=data_root, split="train", download=False, min_images_per_class=min_imgs)
    val_ds = LFWDataset(root=data_root, split="test", download=False, min_images_per_class=min_imgs)

    train_sampler = FewShotEpisodicSampler(
        train_ds,
        n_way=config.n_way,
        k_shot=config.k_shot,
        n_query=config.q_query,
        num_episodes=raw.get("train_episodes", 200),
        class_split="train",
        seed=seed,
    )
    val_sampler = FewShotEpisodicSampler(
        val_ds,
        n_way=config.n_way,
        k_shot=config.k_shot,
        n_query=config.q_query,
        num_episodes=raw.get("val_episodes", 100),
        class_split="test",
        seed=seed + 1,
    )

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=0, collate_fn=collate_episode)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=0, collate_fn=collate_episode)

    # Train -----------------------------------------------------------------------
    backbone = BackboneEmbedder()
    trainer = ProtoNetTrainer(backbone, config, train_loader, val_loader)
    out_dir = Path(out_dir).resolve()
    history = trainer.train(save_dir=str(out_dir))

    best_ckpt = out_dir / "best_checkpoint.pt"
    log.info("Training complete. Best checkpoint: %s", best_ckpt)
    return best_ckpt


def cmd_train_protonet(args: argparse.Namespace) -> None:
    train_protonet(
        config_path=args.config,
        data_root=args.data_root,
        out_dir=args.out_dir,
        device=args.device,
    )


# ── Pipeline step 4 – eval-lfw ───────────────────────────────────────────────


def _generate_report(
    few_shot_rows: list[dict],
    pairs_metrics: Optional[dict],
    output_dir: Path,
) -> Path:
    """
    Write a Markdown summary report and a PDF/PNG figure to *output_dir*.

    The Markdown report contains:
    - Top-1 accuracy per K table
    - TAR@FAR table from pairs evaluation (when available)

    The PNG/PDF figure renders the same tables as a matplotlib figure.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = [
        "# LFW Baseline Evaluation Report",
        "",
        f"Generated: {timestamp}",
        "",
        "---",
        "",
        "## Few-Shot Accuracy per K (N-way K-shot)",
        "",
        "| K (shots) | N-way | Mean Accuracy | ±95% CI | Episodes |",
        "|----------:|------:|--------------:|--------:|---------:|",
    ]
    for row in few_shot_rows:
        report_lines.append(
            f"| {row['k_shot']:>9} | {int(row['n_way']):>5} "
            f"| {row['mean_accuracy']:.4f}        | {row['ci95']:.4f}  "
            f"| {int(row['num_episodes']):>8} |"
        )

    if pairs_metrics:
        report_lines += [
            "",
            "---",
            "",
            "## Pairs Verification Metrics",
            "",
            "| Metric | Value |",
            "|--------|------:|",
        ]
        friendly = {
            "accuracy":          "Accuracy",
            "f1":                "F1 Score",
            "roc_auc":           "ROC AUC",
            "tar_at_far_0.1pct": "TAR @ FAR=0.1%",
            "tar_at_far_1pct":   "TAR @ FAR=1%",
        }
        for key, label in friendly.items():
            if key in pairs_metrics:
                report_lines.append(f"| {label} | {pairs_metrics[key]:.4f} |")

    report_lines += ["", "---", "", "_Privacy Campus Security – Baseline Evaluation_", ""]
    report_md = output_dir / "report.md"
    report_md.write_text("\n".join(report_lines), encoding="utf-8")
    log.info("report.md → %s", report_md)

    # ── PNG / PDF figure ────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        k_vals = [r["k_shot"] for r in few_shot_rows]
        accs   = [r["mean_accuracy"] for r in few_shot_rows]
        ci95s  = [r["ci95"] for r in few_shot_rows]

        n_panels = 2 if pairs_metrics else 1
        fig = plt.figure(figsize=(12, 5 * n_panels))
        gs  = gridspec.GridSpec(n_panels, 1, figure=fig, hspace=0.45)

        # Panel 1 – few-shot bar chart
        ax1 = fig.add_subplot(gs[0])
        bars = ax1.bar(
            [str(k) for k in k_vals], accs,
            yerr=ci95s, capsize=5,
            color="steelblue", alpha=0.85, error_kw={"elinewidth": 1.5},
        )
        ax1.set_ylim(0, 1)
        ax1.set_xlabel("K (shots per class)", fontsize=11)
        ax1.set_ylabel("Mean Accuracy ±95% CI", fontsize=11)
        ax1.set_title("Few-Shot Accuracy per K (LFW)", fontsize=13, fontweight="bold")
        for bar, acc, ci in zip(bars, accs, ci95s):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ci + 0.01,
                f"{acc:.3f}",
                ha="center", va="bottom", fontsize=9,
            )

        # Panel 2 – pairs verification table (if available)
        if pairs_metrics and n_panels > 1:
            ax2 = fig.add_subplot(gs[1])
            ax2.axis("off")
            rows_data = [
                ["Accuracy",       f"{pairs_metrics.get('accuracy', float('nan')):.4f}"],
                ["F1 Score",       f"{pairs_metrics.get('f1', float('nan')):.4f}"],
                ["ROC AUC",        f"{pairs_metrics.get('roc_auc', float('nan')):.4f}"],
                ["TAR @ FAR=0.1%", f"{pairs_metrics.get('tar_at_far_0.1pct', float('nan')):.4f}"],
                ["TAR @ FAR=1%",   f"{pairs_metrics.get('tar_at_far_1pct', float('nan')):.4f}"],
            ]
            table = ax2.table(
                cellText=rows_data,
                colLabels=["Metric", "Value"],
                cellLoc="center",
                loc="center",
                bbox=[0.2, 0.0, 0.6, 1.0],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            ax2.set_title("Pairs Verification Metrics (LFW)", fontsize=13, fontweight="bold")

        fig.suptitle("LFW Baseline Evaluation", fontsize=15, fontweight="bold", y=1.01)

        png_path = output_dir / "report.png"
        fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
        log.info("Report PNG → %s", png_path)

        pdf_path = output_dir / "report.pdf"
        fig.savefig(str(pdf_path), bbox_inches="tight")
        log.info("Report PDF → %s", pdf_path)

        plt.close(fig)

    except ImportError:
        log.warning("matplotlib not installed – PNG/PDF report skipped. Install with: pip install matplotlib")

    return report_md


def eval_lfw(
    embeddings_path: Optional[str],
    episodes: int = 1000,
    k_values: list[int] | None = None,
    n_way: int = 5,
    n_query: int = 15,
    out_dir: str | Path = "experiments/lfw_results",
    checkpoint: Optional[str] = None,
    root: str | Path = "data/lfw",
    device: Optional[str] = None,
    seed: int = 42,
    run_pairs: bool = True,
) -> dict:
    """
    Run full LFW episodic evaluation across multiple K values and produce a report.

    Orchestrates:
    1. Pairs verification (TAR/FAR, ROC-AUC) – skipped when embeddings_path is given
       and *run_pairs* is False.
    2. N-way K-shot accuracy for every K in *k_values*.
    3. Aggregate metrics CSV.
    4. Markdown + PNG/PDF report.

    Args:
        embeddings_path: Path to ``.npz`` or ``.pt`` precomputed embeddings; when
                         ``None`` the backbone runs over raw LFW images.
        episodes:        Number of episodes per K evaluation.
        k_values:        List of K values (default ``[1, 2, 3, 5]``).
        n_way:           Classes per episode.
        n_query:         Query images per class.
        out_dir:         Output directory (created if absent).
        checkpoint:      Optional backbone checkpoint ``*.pt``.
        root:            LFW data root (used when *embeddings_path* is ``None``).
        device:          PyTorch device (auto-detected when ``None``).
        seed:            Global reproducibility seed.
        run_pairs:       Whether to run pairs verification in addition to few-shot.

    Returns:
        Dict containing ``few_shot_rows`` and ``pairs_metrics``.
    """
    import torch

    from src.eval.evaluate_lfw import evaluate_few_shot, evaluate_k_sweep, evaluate_pairs

    torch.manual_seed(seed)

    k_values = k_values or [1, 2, 3, 5]
    device = device or _auto_device()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file = str(out_dir / "eval.log")
    _setup_logging(log_file=log_file)
    log.info("eval-lfw | embeddings=%s | episodes=%d | k=%s | n_way=%d", embeddings_path, episodes, k_values, n_way)

    # Resolve embedding path: support both .npz and .pt ────────────────────────
    emb_path_for_eval: Optional[str] = None
    if embeddings_path:
        ep = Path(embeddings_path)
        if not ep.exists():
            raise FileNotFoundError(f"Embeddings file not found: {ep}")
        if ep.suffix == ".npz":
            # Convert .npz → .pt for compatibility with evaluate_few_shot
            pt_tmp = out_dir / "_embeddings_cache.pt"
            embs, labels, classes = _load_embeddings_npz(ep)
            torch.save({"embeddings": embs, "labels": labels, "classes": classes}, str(pt_tmp))
            emb_path_for_eval = str(pt_tmp)
            log.info("Converted .npz → .pt cache at %s", pt_tmp)
        else:
            emb_path_for_eval = str(ep)

    # ── Pairs verification ────────────────────────────────────────────────────
    pairs_metrics: Optional[dict] = None
    if run_pairs and emb_path_for_eval is None:
        # Pairs eval requires raw images (torchvision LFWPairs)
        try:
            pairs_metrics = evaluate_pairs(
                root=root,
                device=device,
                model_checkpoint=checkpoint,
                output_dir=out_dir,
                save_json=True,
            )
        except Exception as exc:
            log.warning("Pairs evaluation failed (skipping): %s", exc)

    # ── Few-shot K-sweep ──────────────────────────────────────────────────────
    few_shot_rows = evaluate_k_sweep(
        k_values=k_values,
        n_way=n_way,
        num_episodes=episodes,
        output_dir=out_dir,
        root=root,
        n_query=n_query,
        device=device,
        seed=seed,
        model_checkpoint=checkpoint,
        embeddings_path=emb_path_for_eval,
        save_json=True,
    )

    # ── Aggregate CSV ─────────────────────────────────────────────────────────
    csv_path = out_dir / "metrics_summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(few_shot_rows[0].keys()))
        writer.writeheader()
        writer.writerows(few_shot_rows)
    log.info("Metrics CSV → %s", csv_path)

    # Append pairs metrics to CSV if present
    if pairs_metrics:
        pairs_csv_path = out_dir / "pairs_metrics.csv"
        with pairs_csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(pairs_metrics.keys()))
            writer.writeheader()
            writer.writerow(pairs_metrics)
        log.info("Pairs CSV → %s", pairs_csv_path)

    # ── Report ────────────────────────────────────────────────────────────────
    _generate_report(few_shot_rows, pairs_metrics, out_dir)

    return {"few_shot_rows": few_shot_rows, "pairs_metrics": pairs_metrics}


def cmd_eval_lfw(args: argparse.Namespace) -> None:
    eval_lfw(
        embeddings_path=args.embeddings,
        episodes=args.episodes,
        k_values=args.k,
        n_way=args.n_way,
        n_query=args.n_query,
        out_dir=args.out,
        checkpoint=args.checkpoint,
        root=args.root,
        device=args.device,
        seed=args.seed,
        run_pairs=not args.skip_pairs,
    )


# ── Operational sub-commands (enroll / verify / eval / serve) ─────────────────


def cmd_enroll(args: argparse.Namespace) -> None:
    """Enroll a new subject into the embedding store."""
    import torch
    from PIL import Image

    from src.data.lfw_loader import DEFAULT_TRANSFORM
    from src.models.backbone import BackboneEmbedder
    from src.privacy.embedding_store import EmbeddingStore

    store = EmbeddingStore(args.store, encrypt=args.encrypt)
    model = BackboneEmbedder().eval()

    img = DEFAULT_TRANSFORM(Image.open(args.image).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        embedding = model(img).squeeze(0).numpy()

    store.add(args.subject_id, embedding)
    store.save()
    log.info("Enrolled subject '%s'.", args.subject_id)


def cmd_enroll_batch(args: argparse.Namespace) -> None:
    """Detect all faces in *image* and enroll each as a separate anonymous record.

    For every detected face a unique opaque subject key is generated:

        grp:<group_id>:f<face_idx>:<secrets.token_hex(16)>

    The plaintext key is returned to the caller (so they can use it for future
    ``verify`` calls) but *never written to disk* — only its
    ``SHA-256(salt || key)`` hash is stored in the embedding store.
    """
    import json
    import secrets

    from PIL import Image

    from src.models.backbone import BackboneEmbedder
    from src.privacy.embedding_store import EmbeddingStore
    from src.web.face_utils import generate_embedding_from_image

    store = EmbeddingStore(args.store, encrypt=args.encrypt)
    model = BackboneEmbedder().eval()

    pil_img = Image.open(args.image).convert("RGB")
    results = generate_embedding_from_image(pil_img, model=model, device="cpu")

    if not results:
        log.warning("No faces detected in '%s'.", args.image)
        print(json.dumps([]))
        return

    group_id = args.group_id
    enrolled_keys: list[str] = []

    for face in results:
        token = secrets.token_hex(16)
        subject_key = f"grp:{group_id}:f{face['face_id_local']}:{token}"
        store.add(subject_key, face["embedding"])
        enrolled_keys.append(subject_key)

    store.save()
    log.info(
        "Batch enrolled %d face(s) from '%s' into '%s'.",
        len(enrolled_keys), args.image, args.store,
    )
    print(json.dumps(enrolled_keys, indent=2))


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify whether a query image matches a stored subject."""
    import numpy as np
    import torch
    from PIL import Image

    from src.data.lfw_loader import DEFAULT_TRANSFORM
    from src.models.backbone import BackboneEmbedder
    from src.privacy.embedding_store import EmbeddingStore

    store = EmbeddingStore(args.store, encrypt=args.encrypt)
    if args.subject_id not in store:
        log.error("Subject '%s' not found in store.", args.subject_id)
        sys.exit(1)

    model = BackboneEmbedder().eval()
    img = DEFAULT_TRANSFORM(Image.open(args.image).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        query_emb = model(img).squeeze(0).numpy()

    stored_emb = store.get(args.subject_id)
    sim = float(np.dot(query_emb, stored_emb))
    match = sim >= args.threshold
    log.info(
        "Similarity: %.4f | Threshold: %.2f | Match: %s", sim, args.threshold, match
    )


def cmd_eval(args: argparse.Namespace) -> None:
    from src.eval.evaluate_lfw import evaluate_pairs

    evaluate_pairs(root=args.root, batch_size=args.batch_size, device=args.device)


def cmd_serve(args: argparse.Namespace) -> None:
    from src.web.flask_app import create_app

    app = create_app()
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


# ── simulate-camera ───────────────────────────────────────────────────────────


def cmd_simulate_camera(args: argparse.Namespace) -> None:
    """
    Read image files from *folder* and publish each one as a JPEG payload to
    the MQTT topic ``campus/<camera_id>/frame`` at the requested frame-rate.

    Also publishes a lightweight JSON envelope to
    ``campus/<camera_id>/sensor`` with frame metadata (filename, timestamp,
    sequence number) alongside every frame publish.

    MQTT broker settings are resolved in priority order:
    1. ``--broker`` / ``--port`` CLI flags (highest priority)
    2. ``mqtt.broker_host`` / ``mqtt.broker_port`` in ``configs/cameras.yaml``
    3. Defaults: ``localhost:1883``

    If ``--loop`` is given, the folder is replayed in a round-robin loop at the
    requested fps until the user hits Ctrl-C.  Without ``--loop`` the command
    exits after the first pass.
    """
    import io
    import time

    try:
        import paho.mqtt.client as mqtt  # type: ignore[import]
    except ImportError as exc:
        log.error(
            "paho-mqtt is required for simulate-camera.  "
            "Install with:  pip install paho-mqtt"
        )
        raise SystemExit(1) from exc

    from PIL import Image  # type: ignore[import]

    # ── Resolve broker settings ───────────────────────────────────────────────
    broker_host: str = args.broker or "localhost"
    broker_port: int = args.port
    topic_prefix: str = "campus"

    cfg_path = Path(args.config)
    if cfg_path.exists():
        try:
            from src.privacy.camera_store import CameraRegistry
            reg = CameraRegistry(cfg_path)
            if not args.broker:
                broker_host = reg.broker_host
            if args.port == 1883:  # user left default → allow override from YAML
                broker_port = reg.broker_port
            topic_prefix = reg.topic_prefix
            if args.camera_id in reg:
                cam = reg.get(args.camera_id)
                log.info(
                    "Camera: %s (%s) | door=%s | lat=%.4f lon=%.4f",
                    cam.camera_id, cam.name, cam.door_id, cam.lat, cam.lon,
                )
        except Exception as exc:
            log.warning("Could not load camera registry from %s: %s", cfg_path, exc)

    frame_topic  = f"{topic_prefix}/{args.camera_id}/frame"
    sensor_topic = f"{topic_prefix}/{args.camera_id}/sensor"

    # ── Collect image files ───────────────────────────────────────────────────
    _IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    folder = Path(args.folder)
    if not folder.is_dir():
        log.error("Frame folder does not exist: %s", folder)
        raise SystemExit(1)

    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_EXTENSIONS
    )
    if not images:
        log.error(
            "No image files found in %s  "
            "(supported: %s)",
            folder, ", ".join(sorted(_IMG_EXTENSIONS)),
        )
        raise SystemExit(1)

    log.info(
        "simulate-camera | camera=%s | folder=%s | frames=%d | fps=%.2f | loop=%s",
        args.camera_id, folder, len(images), args.fps, args.loop,
    )
    log.info("MQTT broker: %s:%d", broker_host, broker_port)
    log.info("Frame topic:  %s", frame_topic)
    log.info("Sensor topic: %s", sensor_topic)

    # ── MQTT client ───────────────────────────────────────────────────────────
    # paho-mqtt ≥ 2.0 requires an explicit callback API version.
    try:
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"pcs-sim-{args.camera_id}",
            clean_session=True,
        )
    except AttributeError:
        # Older paho-mqtt (< 2.0) – fall back to legacy constructor
        client = mqtt.Client(client_id=f"pcs-sim-{args.camera_id}", clean_session=True)  # type: ignore[call-arg]

    client.connect(broker_host, broker_port, keepalive=60)
    client.loop_start()

    sleep_s = 1.0 / max(args.fps, 0.001)
    sequence = 0
    passes = 0

    def _publish_frame(img_path: Path, seq: int) -> None:
        """Convert image to JPEG bytes and publish both topics."""
        # Re-encode as JPEG regardless of source format to normalise payload
        img = Image.open(img_path).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=args.quality)
        jpeg_bytes = buf.getvalue()

        ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # ── Frame payload (raw bytes) ─────────────────────────────────────────
        result = client.publish(frame_topic, jpeg_bytes, qos=args.qos)
        result.wait_for_publish()

        # ── Sensor metadata (JSON string) ─────────────────────────────────────
        sensor_payload = json.dumps({
            "camera_id":  args.camera_id,
            "seq":        seq,
            "filename":   img_path.name,
            "timestamp":  ts,
            "width":      img.width,
            "height":     img.height,
            "fps":        args.fps,
            "source":     "simulated",
        })
        client.publish(sensor_topic, sensor_payload, qos=args.qos)

        log.info(
            "[seq=%05d] published %s → %s  (%d bytes)",
            seq, img_path.name, frame_topic, len(jpeg_bytes),
        )

    try:
        while True:
            passes += 1
            for img_path in images:
                _publish_frame(img_path, sequence)
                sequence += 1
                time.sleep(sleep_s)

            if not args.loop:
                break
            log.info("Pass %d complete – looping (Ctrl-C to stop) …", passes)

    except KeyboardInterrupt:
        log.info("simulate-camera stopped by user after %d frame(s).", sequence)
    finally:
        client.loop_stop()
        client.disconnect()
        log.info("MQTT client disconnected.  Total frames published: %d", sequence)


# ── simulate-door ─────────────────────────────────────────────────────────────


def cmd_simulate_door(args: argparse.Namespace) -> None:
    """
    Publish synthetic door-sensor events to ``campus/<door_id>/event``  on the
    MQTT broker.

    Each published message has a UTF-8 JSON body of the form::

        {
            "type":      "open" | "close",
            "door_id":   "<door_id>",
            "timestamp": "<ISO-8601 UTC>"
        }

    This is the canonical schema consumed by the Flask server's MQTT subscriber
    which feeds events into the ``AccessControlEngine``.

    Resolution order for broker settings:
    1. ``--broker`` / ``--port`` CLI flags
    2. ``mqtt.broker_host`` / ``mqtt.broker_port`` in ``configs/cameras.yaml``
    3. Defaults: ``localhost:1883``
    """
    import time

    try:
        import paho.mqtt.client as mqtt  # type: ignore[import]
    except ImportError as exc:
        log.error(
            "paho-mqtt is required for simulate-door.  "
            "Install with:  pip install paho-mqtt"
        )
        raise SystemExit(1) from exc

    # ── Resolve broker settings ───────────────────────────────────────────────
    broker_host: str = args.broker or "localhost"
    broker_port: int = args.port
    topic_prefix: str = "campus"

    cfg_path = Path(args.config)
    if cfg_path.exists():
        try:
            from src.privacy.camera_store import CameraRegistry
            reg = CameraRegistry(cfg_path)
            if not args.broker:
                broker_host = reg.broker_host
            if args.port == 1883:
                broker_port = reg.broker_port
            topic_prefix = reg.topic_prefix
        except Exception as exc:
            log.warning("Could not load camera registry from %s: %s", cfg_path, exc)

    door_topic = f"{topic_prefix}/{args.door_id}/event"

    # ── Parse pattern ─────────────────────────────────────────────────────────
    valid_types = {"open", "close"}
    pattern = [t.strip().lower() for t in args.pattern.split(",")]
    for t in pattern:
        if t not in valid_types:
            log.error("Invalid event type '%s'. Must be one of: %s", t, valid_types)
            raise SystemExit(1)
    if not pattern:
        log.error("--pattern must specify at least one event type.")
        raise SystemExit(1)

    log.info(
        "simulate-door | door=%s | pattern=%s | interval=%.1fs | count=%d",
        args.door_id, pattern, args.interval, args.count,
    )
    log.info("MQTT broker: %s:%d  topic: %s", broker_host, broker_port, door_topic)

    # ── MQTT client ───────────────────────────────────────────────────────────
    try:
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"pcs-door-{args.door_id}",
            clean_session=True,
        )
    except AttributeError:
        client = mqtt.Client(client_id=f"pcs-door-{args.door_id}", clean_session=True)  # type: ignore[call-arg]

    client.connect(broker_host, broker_port, keepalive=60)
    client.loop_start()

    published = 0
    pattern_len = len(pattern)

    try:
        while True:
            event_type = pattern[published % pattern_len]
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            payload = json.dumps({
                "type":      event_type,
                "door_id":   args.door_id,
                "timestamp": ts,
            })
            result = client.publish(door_topic, payload, qos=args.qos)
            result.wait_for_publish()
            log.info(
                "[event=%04d] %s → %s  (%s)",
                published + 1, event_type.upper(), door_topic, ts,
            )
            published += 1

            if args.count > 0 and published >= args.count:
                break

            # Only sleep if we have more events to publish
            if args.count == 0 or published < args.count:
                time.sleep(args.interval)

    except KeyboardInterrupt:
        log.info("simulate-door stopped by user.")
    finally:
        client.loop_stop()
        client.disconnect()
        log.info("simulate-door finished.  Total events published: %d", published)


# ── Argument parsing ──────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcs",
        description="Privacy Campus Security CLI – baseline pipeline & operational commands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.strip(),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── prepare-data ─────────────────────────────────────────────────────────
    p_prep = sub.add_parser(
        "prepare-data",
        help="Download & verify LFW (or other) dataset",
        description="Step 1 – Download and verify the target dataset.",
    )
    p_prep.add_argument("--dataset", default="lfw", choices=["lfw"],
                        help="Dataset name (default: lfw)")
    p_prep.add_argument("--out-dir", default="data/lfw",
                        help="Root directory for downloaded data (default: data/lfw)")
    p_prep.set_defaults(func=cmd_prepare_data)

    # ── compute-embeddings ────────────────────────────────────────────────────
    p_emb = sub.add_parser(
        "compute-embeddings",
        help="Run backbone over LFW and save .npz embeddings",
        description="Step 2 – Compute L2-normalised embeddings for every LFW image.",
    )
    p_emb.add_argument("--root", default="data/lfw",
                       help="LFW data root (default: data/lfw)")
    p_emb.add_argument("--checkpoint", default=None,
                       help="Optional ProtoNetTrainer .pt checkpoint path")
    p_emb.add_argument("--out", default="embeddings/lfw_embeddings.npz",
                       help="Output .npz path (default: embeddings/lfw_embeddings.npz)")
    p_emb.add_argument("--batch-size", type=int, default=64,
                       help="Inference batch size (default: 64)")
    p_emb.add_argument("--device", default=None,
                       help="cpu / cuda (auto-detected when omitted)")
    p_emb.set_defaults(func=cmd_compute_embeddings)

    # ── train-protonet ────────────────────────────────────────────────────────
    p_train = sub.add_parser(
        "train-protonet",
        help="Train a Prototypical Network via episodic meta-learning",
        description="Step 3 – Load configs/proto.yaml and run ProtoNetTrainer.",
    )
    p_train.add_argument("--config", default="configs/proto.yaml",
                         help="Path to proto.yaml config (default: configs/proto.yaml)")
    p_train.add_argument("--data-root", default="data/lfw",
                         help="LFW data root (default: data/lfw)")
    p_train.add_argument("--out-dir", default="experiments/checkpoints",
                         help="Directory for checkpoints (default: experiments/checkpoints)")
    p_train.add_argument("--device", default=None,
                         help="Override device from config (cpu/cuda)")
    p_train.set_defaults(func=cmd_train_protonet)

    # ── eval-lfw ─────────────────────────────────────────────────────────────
    p_ev = sub.add_parser(
        "eval-lfw",
        help="Run episodic evaluation for multiple K; produce CSV + report",
        description="Step 4 – Aggregate few-shot metrics and generate a PDF/PNG report.",
    )
    p_ev.add_argument("--embeddings", default=None,
                      help="Path to precomputed .npz/.pt embeddings (skips backbone inference)")
    p_ev.add_argument("--episodes", type=int, default=1000,
                      help="Episodes per K value (default: 1000)")
    p_ev.add_argument("--k", nargs="+", type=int, default=[1, 2, 3, 5],
                      metavar="K",
                      help="Shot counts to evaluate, e.g. --k 1 2 3 5 (default: 1 2 3 5)")
    p_ev.add_argument("--n-way", type=int, default=5,
                      help="Classes per episode (default: 5)")
    p_ev.add_argument("--n-query", type=int, default=15,
                      help="Query images per class (default: 15)")
    p_ev.add_argument("--out", default="experiments/lfw_results",
                      help="Output directory (default: experiments/lfw_results)")
    p_ev.add_argument("--checkpoint", default=None,
                      help="Optional backbone checkpoint .pt (ignored when --embeddings is set)")
    p_ev.add_argument("--root", default="data/lfw",
                      help="LFW data root for raw-image mode (default: data/lfw)")
    p_ev.add_argument("--device", default=None,
                      help="cpu / cuda (auto-detected when omitted)")
    p_ev.add_argument("--seed", type=int, default=42,
                      help="Reproducibility seed (default: 42)")
    p_ev.add_argument("--skip-pairs", action="store_true",
                      help="Skip pairs verification (TAR/FAR) even when raw images are used")
    p_ev.set_defaults(func=cmd_eval_lfw)

    # ── enroll ────────────────────────────────────────────────────────────────
    p_enroll = sub.add_parser("enroll", help="Enroll a subject into the embedding store")
    p_enroll.add_argument("--subject-id", required=True)
    p_enroll.add_argument("--image", required=True)
    p_enroll.add_argument("--store", default="data/embeddings.json")
    p_enroll.add_argument("--encrypt", action="store_true")
    p_enroll.set_defaults(func=cmd_enroll)

    # ── enroll-batch ──────────────────────────────────────────────────────────
    p_enroll_batch = sub.add_parser(
        "enroll-batch",
        help="Detect all faces in an image and enroll each as an anonymous record",
        description=(
            "Detect every face in the supplied image and enroll them as separate,\n"
            "anonymised records. Each face receives a salted opaque key of the\n"
            "form 'grp:<group_id>:f<idx>:<random_token>'; only the SHA-256 hash\n"
            "of that key is written to disk. The plaintext keys are printed to\n"
            "stdout as a JSON list for the caller to retain."
        ),
    )
    p_enroll_batch.add_argument("--image", required=True,
                                help="Path to the input image file")
    p_enroll_batch.add_argument(
        "--group-id", default="default", metavar="GROUP_ID",
        help="Group tag embedded in every generated subject key (default: default)",
    )
    p_enroll_batch.add_argument("--store", default="data/embeddings.json",
                                help="Path to the embedding store (default: data/embeddings.json)")
    p_enroll_batch.add_argument("--encrypt", action="store_true",
                                help="Enable Fernet at-rest encryption")
    p_enroll_batch.set_defaults(func=cmd_enroll_batch)

    # ── verify ────────────────────────────────────────────────────────────────
    p_verify = sub.add_parser("verify", help="Verify a subject against the embedding store")
    p_verify.add_argument("--subject-id", required=True)
    p_verify.add_argument("--image", required=True)
    p_verify.add_argument("--store", default="data/embeddings.json")
    p_verify.add_argument("--encrypt", action="store_true")
    p_verify.add_argument("--threshold", type=float, default=0.0,
                          help="Cosine similarity threshold (default: 0.0)")
    p_verify.set_defaults(func=cmd_verify)

    # ── eval (pairs only, legacy) ─────────────────────────────────────────────
    p_eval = sub.add_parser("eval", help="Evaluate backbone on LFW pairs (legacy)")
    p_eval.add_argument("--root", default="data/lfw")
    p_eval.add_argument("--batch-size", type=int, default=64)
    p_eval.add_argument("--device", default=None)
    p_eval.set_defaults(func=cmd_eval)

    # ── serve ─────────────────────────────────────────────────────────────────
    p_serve = sub.add_parser("serve", help="Launch the Flask web dashboard")
    p_serve.add_argument("--port", type=int, default=5000)
    p_serve.add_argument("--debug", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    # ── simulate-camera ───────────────────────────────────────────────────────
    p_sim = sub.add_parser(
        "simulate-camera",
        help="Publish image frames from a folder to an MQTT broker at a fixed fps",
        description=(
            "Reads every image file (JPEG/PNG/BMP/WebP) in FOLDER in sorted order\n"
            "and publishes each one as a JPEG byte payload to the MQTT topic\n"
            "  campus/<CAMERA_ID>/frame\n"
            "with a companion JSON metadata message on\n"
            "  campus/<CAMERA_ID>/sensor\n\n"
            "Broker host/port defaults come from configs/cameras.yaml (mqtt section)\n"
            "and are overridden by --broker / --port when supplied.\n\n"
            "Example:\n"
            "  python -m src.cli.run simulate-camera \\\n"
            "      --camera-id gate-north --folder samples/gate_frames --fps 2"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_sim.add_argument(
        "--camera-id", required=True, metavar="CAMERA_ID",
        help="Camera identifier (must match an entry in cameras.yaml for full metadata)",
    )
    p_sim.add_argument(
        "--folder", required=True, metavar="FOLDER",
        help="Directory containing the JPEG/PNG frames to publish",
    )
    p_sim.add_argument(
        "--fps", type=float, default=1.0, metavar="FPS",
        help="Target publish rate in frames-per-second (default: 1.0)",
    )
    p_sim.add_argument(
        "--broker", default=None, metavar="HOST",
        help="MQTT broker hostname (overrides cameras.yaml; default: localhost)",
    )
    p_sim.add_argument(
        "--port", type=int, default=1883, metavar="PORT",
        help="MQTT broker port (overrides cameras.yaml; default: 1883)",
    )
    p_sim.add_argument(
        "--qos", type=int, default=0, choices=[0, 1, 2],
        help="MQTT QoS level for published messages (default: 0)",
    )
    p_sim.add_argument(
        "--quality", type=int, default=85, metavar="QUALITY",
        help="JPEG re-encode quality 1-95 (default: 85)",
    )
    p_sim.add_argument(
        "--loop", action="store_true",
        help="Loop through the folder indefinitely until Ctrl-C",
    )
    p_sim.add_argument(
        "--config", default="configs/cameras.yaml", metavar="YAML",
        help="Path to cameras.yaml used for broker defaults (default: configs/cameras.yaml)",
    )
    p_sim.set_defaults(func=cmd_simulate_camera)

    # ── simulate-door ─────────────────────────────────────────────────────────
    p_door = sub.add_parser(
        "simulate-door",
        help="Publish door sensor events (open/close) to an MQTT broker",
        description=(
            "Publishes synthetic door sensor events to the MQTT topic\n"
            "  campus/<DOOR_ID>/event\n"
            "with payload  {type: 'open'|'close', door_id, timestamp}\n\n"
            "Useful for testing the access-control engine: combine with a\n"
            "failed /api/verify/frame call to trigger a forced_entry alert,\n"
            "or with two rapid verify results for the same door to trigger\n"
            "a tailgate alert.\n\n"
            "Example:\n"
            "  python -m src.cli.run simulate-door \\\n"
            "      --door-id door-01 --pattern open,close --interval 2"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_door.add_argument(
        "--door-id", required=True, metavar="DOOR_ID",
        help="Door identifier, e.g. door-01 (matches door_id in cameras.yaml)",
    )
    p_door.add_argument(
        "--pattern", default="open,close", metavar="EVENTS",
        help=(
            "Comma-separated sequence of event types to cycle through.  "
            "Each value must be 'open' or 'close'.  "
            "The sequence repeats if --loop is set.  "
            "Default: 'open,close'"
        ),
    )
    p_door.add_argument(
        "--interval", type=float, default=3.0, metavar="SECONDS",
        help="Seconds between consecutive events in the pattern (default: 3.0)",
    )
    p_door.add_argument(
        "--count", type=int, default=0, metavar="N",
        help=(
            "Total number of events to publish (0 = unlimited / until Ctrl-C).  "
            "The pattern repeats as needed.  Default: 0"
        ),
    )
    p_door.add_argument(
        "--broker", default=None, metavar="HOST",
        help="MQTT broker hostname (overrides cameras.yaml; default: localhost)",
    )
    p_door.add_argument(
        "--port", type=int, default=1883, metavar="PORT",
        help="MQTT broker port (default: 1883)",
    )
    p_door.add_argument(
        "--qos", type=int, default=0, choices=[0, 1, 2],
        help="MQTT QoS level (default: 0)",
    )
    p_door.add_argument(
        "--config", default="configs/cameras.yaml", metavar="YAML",
        help="Path to cameras.yaml for broker defaults (default: configs/cameras.yaml)",
    )
    p_door.set_defaults(func=cmd_simulate_door)

    return parser


def main() -> None:
    _setup_logging()
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
