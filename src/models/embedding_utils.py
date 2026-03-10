"""
embedding_utils.py – Utilities for batch-computing and caching embeddings.

Typical usage::

    from src.models.backbone import BackboneEmbedder
    from src.models.embedding_utils import save_embeddings
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder("data/lfw", transform=transform)
    model   = BackboneEmbedder().eval()
    save_embeddings(dataset, model, "experiments/sample_runs/embeddings.pt")

The saved ``.pt`` file is a ``dict`` with keys:

* ``"embeddings"`` – ``Tensor`` of shape ``(N, 512)``
* ``"labels"``     – ``Tensor`` of shape ``(N,)`` (integer class indices)
* ``"classes"``    – ``list[str]`` of class names (optional, ``[]`` if unavailable)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def save_embeddings(
    dataset: Dataset,
    model: torch.nn.Module,
    out_path: Union[str, Path],
    batch_size: int = 64,
    num_workers: int = 0,
    device: Optional[torch.device] = None,  # type: ignore[name-defined]
) -> dict[str, object]:
    """
    Compute L2-normalised embeddings for every sample in *dataset* and save them.

    The model is moved to *device* for inference and moved back afterwards.
    Gradient computation is disabled throughout.

    Args:
        dataset:     Any ``torch.utils.data.Dataset`` whose ``__getitem__``
                     returns ``(image_tensor, label)`` or
                     ``(image_tensor, label, *extras)``.  Only the first two
                     elements are used.
        model:       A callable ``nn.Module`` that accepts a batched image
                     tensor and returns a 2-D embedding tensor.
                     :class:`~src.models.backbone.BackboneEmbedder` is the
                     primary intended consumer.
        out_path:    Path at which to save the result as a ``torch.save``
                     checkpoint (``*.pt``).  Parent directories are created
                     automatically.
        batch_size:  Samples per inference batch (default 64).
        num_workers: DataLoader worker processes. Use 0 on Windows to avoid
                     multiprocessing fork issues.
        device:      Torch device for inference.  Defaults to CUDA if
                     available, otherwise CPU.

    Returns:
        A ``dict`` with keys ``"embeddings"``, ``"labels"``, ``"classes"``
        mirroring what is written to *out_path*.

    Raises:
        ValueError: If the dataset appears empty.
    """
    if len(dataset) == 0:  # type: ignore[arg-type]
        raise ValueError("dataset is empty — nothing to embed.")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    model = model.to(device)
    model.eval()

    all_embeddings: list[Tensor] = []
    all_labels: list[Tensor] = []

    logger.info(
        "Computing embeddings for %d samples on %s (batch_size=%d) …",
        len(dataset),  # type: ignore[arg-type]
        device,
        batch_size,
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images: Tensor = batch[0].to(device)
            labels: Tensor = batch[1]
            emb: Tensor = model(images).cpu()
            all_embeddings.append(emb)
            all_labels.append(labels)
            if (batch_idx + 1) % 10 == 0:
                logger.debug("  processed %d / %d batches", batch_idx + 1, len(loader))

    embeddings: Tensor = torch.cat(all_embeddings, dim=0)   # (N, D)
    labels_tensor: Tensor = torch.cat(all_labels, dim=0)    # (N,)

    # Optionally preserve class names when the dataset exposes them.
    classes: list[str] = getattr(dataset, "classes", [])

    payload: dict[str, object] = {
        "embeddings": embeddings,
        "labels": labels_tensor,
        "classes": classes,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    logger.info(
        "Saved %d embeddings (dim=%d) → %s",
        embeddings.shape[0],
        embeddings.shape[1],
        out_path,
    )
    return payload



