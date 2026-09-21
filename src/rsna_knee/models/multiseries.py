"""Study-level multiseries image classifier (DINOv2 or MRI-CORE)."""

from __future__ import annotations

from typing import Any

from rsna_knee.constants import NUM_LABELS
from rsna_knee.models.backbone import BACKBONE_DIMS, create_image_encoder
from rsna_knee.models.pooling import create_attention_pool


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


def create_multiseries_model(
    backbone_name: str = "dinov2_vits14",
    *,
    weights_path: str | None = None,
    freeze_backbone: bool = False,
    pretrained: bool = True,
    num_planes: int = 4,
    dropout: float = 0.1,
    image_size: int | None = None,
    encode_chunk_size: int = 0,
):
    torch, nn = _torch()
    encoder_kwargs: dict[str, Any] = {
        "weights_path": weights_path,
        "freeze": freeze_backbone,
        "pretrained": pretrained,
    }
    if image_size is not None:
        encoder_kwargs["image_size"] = image_size
    encoder = create_image_encoder(backbone_name, **encoder_kwargs)
    dim = getattr(encoder, "embed_dim", BACKBONE_DIMS.get(backbone_name, 384))
    # MRI-CORE ViT-B at 384 is heavy; chunk slice encoding unless overridden.
    if encode_chunk_size <= 0 and backbone_name == "mri_core_vitb":
        encode_chunk_size = 8

    class MultiSeriesStudyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.plane_emb = nn.Embedding(num_planes, dim)
            self.fluid_emb = nn.Embedding(2, dim)
            self.fat_emb = nn.Embedding(2, dim)
            self.slice_pool = create_attention_pool(dim)
            self.series_pool = create_attention_pool(dim)
            self.head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.classifiers = nn.ModuleList([nn.Linear(dim, 1) for _ in range(NUM_LABELS)])
            self.encode_chunk_size = int(encode_chunk_size)

        def encode_slices(self, images: Any) -> Any:
            # images: (B, S, N, 3, H, W)
            b, s, n, c, h, w = images.shape
            flat = images.reshape(b * s * n, c, h, w)
            if self.encode_chunk_size > 0 and flat.shape[0] > self.encode_chunk_size:
                chunks = []
                for start in range(0, flat.shape[0], self.encode_chunk_size):
                    chunks.append(self.encoder(flat[start : start + self.encode_chunk_size]))
                feats = torch.cat(chunks, dim=0)
            else:
                feats = self.encoder(flat)
            return feats.reshape(b, s, n, -1)

        def forward(
            self,
            images: Any,
            plane_ids: Any,
            fluid: Any,
            fat_sup: Any,
            series_mask: Any,
            slice_mask: Any,
        ) -> Any:
            # images: (B,S,N,3,H,W)
            token = self.encode_slices(images)  # (B,S,N,D)
            b, s, n, d = token.shape

            plane = self.plane_emb(plane_ids.clamp(0, num_planes - 1))  # (B,S,D)
            fluid_e = self.fluid_emb(fluid.long().clamp(0, 1))
            fat_e = self.fat_emb(fat_sup.long().clamp(0, 1))
            meta = (plane + fluid_e + fat_e).unsqueeze(2)  # (B,S,1,D)
            token = token + meta

            # Pool slices per series.
            token_s = token.reshape(b * s, n, d)
            mask_s = slice_mask.reshape(b * s, n)
            series_vec = self.slice_pool(token_s, mask_s).reshape(b, s, d)

            # Pool series → study.
            study_vec = self.series_pool(series_vec, series_mask)
            shared = self.head(study_vec)
            logits = torch.cat([clf(shared) for clf in self.classifiers], dim=-1)
            return logits

    return MultiSeriesStudyModel()
