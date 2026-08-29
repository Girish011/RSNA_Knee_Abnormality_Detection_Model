"""Study-level multiseries DINOv2 classifier."""

from __future__ import annotations

from typing import Any

from rsna_knee.constants import NUM_LABELS
from rsna_knee.models.backbone import DINOV2_DIMS, create_dinov2_encoder
from rsna_knee.models.plane_routing import plane_prior_bias_matrix
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
    label_plane_routing: bool = False,
):
    torch, nn = _torch()
    encoder = create_dinov2_encoder(
        backbone_name,
        weights_path=weights_path,
        freeze=freeze_backbone,
        pretrained=pretrained,
    )
    dim = getattr(encoder, "embed_dim", DINOV2_DIMS.get(backbone_name, 384))

    class MultiSeriesStudyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.plane_emb = nn.Embedding(num_planes, dim)
            self.fluid_emb = nn.Embedding(2, dim)
            self.fat_emb = nn.Embedding(2, dim)
            self.slice_pool = create_attention_pool(dim)
            self.series_pool = create_attention_pool(dim)
            self.label_plane_routing = label_plane_routing
            if label_plane_routing:
                bias = plane_prior_bias_matrix()
                self.register_buffer(
                    "plane_prior_bias",
                    torch.as_tensor(bias, dtype=torch.float32),
                    persistent=True,
                )
                self.series_score = nn.Linear(dim, 1)
            self.head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.classifiers = nn.ModuleList([nn.Linear(dim, 1) for _ in range(NUM_LABELS)])

        def encode_slices(self, images: Any) -> Any:
            # images: (B, S, N, 3, H, W)
            b, s, n, c, h, w = images.shape
            flat = images.reshape(b * s * n, c, h, w)
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
            import torch.nn.functional as F

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

            if self.label_plane_routing:
                base_scores = self.series_score(series_vec).squeeze(-1)  # (B,S)
                logits_list = []
                for li, clf in enumerate(self.classifiers):
                    plane_bonus = self.plane_prior_bias[li][plane_ids.clamp(0, num_planes - 1)]
                    scores = base_scores + plane_bonus
                    scores = scores.masked_fill(series_mask <= 0, -1e4)
                    weights = F.softmax(scores, dim=-1).unsqueeze(-1)
                    study_vec = (series_vec * weights).sum(dim=1)
                    shared = self.head(study_vec)
                    logits_list.append(clf(shared))
                logits = torch.cat(logits_list, dim=-1)
                return logits

            # Pool series → study (shared routing).
            study_vec = self.series_pool(series_vec, series_mask)
            shared = self.head(study_vec)
            logits = torch.cat([clf(shared) for clf in self.classifiers], dim=-1)
            return logits

    return MultiSeriesStudyModel()
