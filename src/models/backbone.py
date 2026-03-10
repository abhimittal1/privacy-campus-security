"""
backbone.py – Flexible ResNet-50 backbone for 512-D embedding extraction.

Default architecture:
    ResNet-50 (ImageNet-pretrained, fc removed)
    → Flatten
    → Linear(2048, hidden_dim) → BatchNorm1d → ReLU
    → Linear(hidden_dim, 512)
    → L2 normalise

Optional ArcFace weights can be loaded via :meth:`BackboneEmbedder.load_arcface_weights`.
See that method's docstring for instructions on obtaining and supplying weights.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models
from torchvision.models import ResNet50_Weights

logger = logging.getLogger(__name__)

EMBEDDING_DIM: int = 512
_RESNET50_POOL_DIM: int = 2048  # channels after AdaptiveAvgPool2d


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------


class ProjectionHead(nn.Module):
    """
    Two-layer MLP projection: Flatten → Linear → BatchNorm1d → ReLU → Linear.

    Args:
        in_features:  Dimensionality of the backbone feature vector.
        hidden_dim:   Width of the hidden layer (default = same as *in_features*).
        out_features: Output embedding dimension (default 512).
    """

    def __init__(
        self,
        in_features: int = _RESNET50_POOL_DIM,
        hidden_dim: int = _RESNET50_POOL_DIM,
        out_features: int = EMBEDDING_DIM,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_features, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102
        return self.net(x)


# ---------------------------------------------------------------------------
# Main backbone
# ---------------------------------------------------------------------------


class BackboneEmbedder(nn.Module):
    """
    ResNet-50 backbone with a customizable projection head.

    The final ``fc`` layer of ResNet-50 is replaced by :class:`ProjectionHead`,
    producing L2-normalized 512-D embeddings suitable for cosine-similarity
    matching and prototypical few-shot classification.

    Args:
        embedding_dim: Output embedding dimensionality (default 512).
        hidden_dim:    Width of the intermediate projection layer.
                       Defaults to 2048 (same as ResNet-50 pool output).
        freeze_base:   If ``True``, only the projection head is trained;
                       ResNet-50 body weights are frozen.
        pretrained:    Load ImageNet-1k pre-trained weights for ResNet-50.
                       Pass ``False`` for random initialisation (debugging/CI).

    Example::

        model = BackboneEmbedder(freeze_base=False)
        emb = model(torch.randn(4, 3, 224, 224))  # → (4, 512)
    """

    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM,
        hidden_dim: int = _RESNET50_POOL_DIM,
        freeze_base: bool = False,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Keep every layer except the classification head.
        # After this the output shape is (B, 2048, 1, 1).
        self.body = nn.Sequential(*list(resnet.children())[:-1])
        in_features: int = resnet.fc.in_features  # 2048 for ResNet-50

        if freeze_base:
            for param in self.body.parameters():
                param.requires_grad = False

        self.projector = ProjectionHead(
            in_features=in_features,
            hidden_dim=hidden_dim,
            out_features=embedding_dim,
        )

        self.embedding_dim = embedding_dim

    # ------------------------------------------------------------------
    # ArcFace weight loading
    # ------------------------------------------------------------------

    def load_arcface_weights(self, path: Optional[str] = None) -> None:
        """
        Optionally initialise the backbone with ArcFace-pretrained weights.

        **How to supply weights**

        1. Download a ResNet-50 ArcFace checkpoint trained on MS1MV2 or
           similar from one of the following sources:

           * InsightFace model zoo:
             https://github.com/deepinsight/insightface/tree/master/model_zoo
             (look for ``r50`` or ``glint360k_r50`` weights)
           * Pre-converted PyTorch state-dicts from community mirrors.

        2. The checkpoint must be a ``torch.save``'d ``state_dict`` whose
           keys match a standard ``torchvision.models.resnet50`` *without* the
           ``fc`` layer, **or** the full checkpoint with ``fc.*`` keys (those
           will be silently ignored via ``strict=False``).

        3. Pass the path as the *path* argument::

               model.load_arcface_weights("weights/arcface_r50_ms1mv2.pth")

        If *path* is ``None`` (default) a warning is logged and the method
        returns immediately — the model keeps its ImageNet-pretrained or
        randomly initialised weights.

        Args:
            path: Filesystem path to a ``.pth`` / ``.pt`` checkpoint file.
                  ``None`` is a no-op (with a logged warning).

        Raises:
            FileNotFoundError: If *path* is provided but does not exist.
        """
        if path is None:
            warnings.warn(
                "load_arcface_weights called without a path — "
                "keeping current backbone weights.",
                UserWarning,
                stacklevel=2,
            )
            return

        ckpt_path = Path(path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"ArcFace checkpoint not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location="cpu")
        # Unwrap common checkpoint wrappers.
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model", state))

        # Strip common prefixes so keys align with self.body.
        cleaned: dict[str, Tensor] = {}
        for k, v in state.items():
            key = k.removeprefix("module.").removeprefix("body.")
            cleaned[key] = v

        missing, unexpected = self.body.load_state_dict(cleaned, strict=False)
        logger.info(
            "ArcFace weights loaded from %s | missing=%d unexpected=%d",
            ckpt_path,
            len(missing),
            len(unexpected),
        )
        if missing:
            logger.debug("Missing keys: %s", missing[:10])
        if unexpected:
            logger.debug("Unexpected keys: %s", unexpected[:10])

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute L2-normalised 512-D embedding.

        Args:
            x: Input image tensor of shape ``(B, 3, H, W)``.
               Recommended input size: 224 × 224 (ImageNet) or 112 × 112 (ArcFace).

        Returns:
            Tensor of shape ``(B, embedding_dim)`` with unit L2 norm per row.
        """
        features: Tensor = self.body(x)       # (B, 2048, 1, 1)
        emb: Tensor = self.projector(features)  # (B, 512)
        return F.normalize(emb, p=2, dim=1)
