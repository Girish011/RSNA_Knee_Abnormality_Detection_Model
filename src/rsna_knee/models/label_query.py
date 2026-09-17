"""Order-aware, plane-routed label-query aggregation for knee MRI studies."""

from __future__ import annotations

from typing import Any

from rsna_knee.constants import LABEL_COLS, LABEL_PLANE_PRIOR, NUM_LABELS, PLANE_TO_ID
from rsna_knee.models.pooling import create_attention_pool


def _torch():
    import torch
    import torch.nn as nn

    return torch, nn


def create_label_query_model(
    *,
    encoder: Any | None = None,
    input_dim: int | None = None,
    hidden_dim: int = 384,
    num_heads: int = 6,
    sequence_layers: int = 1,
    dropout: float = 0.1,
    plane_prior_penalty: float = 2.0,
):
    """Create a model accepting either images (with encoder) or cached slice features."""
    torch, nn = _torch()
    if input_dim is None:
        input_dim = int(getattr(encoder, "embed_dim", hidden_dim))
    if encoder is None and input_dim is None:
        raise ValueError("Provide an encoder or input_dim")
    if hidden_dim % num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads")

    prior = torch.zeros(NUM_LABELS, len(PLANE_TO_ID), dtype=torch.float32)
    for label_idx, label in enumerate(LABEL_COLS):
        preferred = {PLANE_TO_ID[name] for name in LABEL_PLANE_PRIOR[label]}
        for plane_idx in range(len(PLANE_TO_ID)):
            if plane_idx not in preferred:
                prior[label_idx, plane_idx] = -float(plane_prior_penalty)

    class LabelQueryStudyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.input_dim = int(input_dim)
            self.hidden_dim = hidden_dim
            self.input_proj = nn.Linear(self.input_dim, hidden_dim)
            self.plane_emb = nn.Embedding(len(PLANE_TO_ID), hidden_dim)
            self.fluid_emb = nn.Embedding(2, hidden_dim)
            self.fat_emb = nn.Embedding(2, hidden_dim)
            self.position_mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.series_encoder = nn.TransformerEncoder(layer, num_layers=sequence_layers)
            self.label_queries = nn.Parameter(torch.empty(NUM_LABELS, hidden_dim))
            nn.init.trunc_normal_(self.label_queries, std=0.02)
            self.cross_attention = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.output_norm = nn.LayerNorm(hidden_dim)
            self.classifier_weight = nn.Parameter(torch.empty(NUM_LABELS, hidden_dim))
            self.classifier_bias = nn.Parameter(torch.zeros(NUM_LABELS))
            nn.init.trunc_normal_(self.classifier_weight, std=0.02)
            self.register_buffer("label_plane_prior", prior, persistent=True)

        def encode_images(self, images: Any) -> Any:
            if self.encoder is None:
                raise ValueError("This model expects cached features, not images")
            b, s, n, c, h, w = images.shape
            flat = images.reshape(b * s * n, c, h, w)
            encoded = self.encoder(flat)
            return encoded.reshape(b, s, n, -1)

        def forward(
            self,
            inputs: Any,
            plane_ids: Any,
            fluid: Any,
            fat_sup: Any,
            series_mask: Any,
            slice_mask: Any,
            slice_positions: Any,
            return_embeddings: bool = False,
        ) -> Any:
            features = self.encode_images(inputs) if inputs.ndim == 7 else inputs
            if features.ndim != 4:
                raise ValueError(f"Expected (B,S,N,D) features, got {features.shape}")
            b, s, n, _ = features.shape
            token = self.input_proj(features)
            meta = (
                self.plane_emb(plane_ids.long().clamp(0, len(PLANE_TO_ID) - 1))
                + self.fluid_emb(fluid.long().clamp(0, 1))
                + self.fat_emb(fat_sup.long().clamp(0, 1))
            )
            token = token + meta.unsqueeze(2) + self.position_mlp(slice_positions.unsqueeze(-1))

            token_s = token.reshape(b * s, n, self.hidden_dim)
            valid_s = slice_mask.reshape(b * s, n) > 0
            # PyTorch attention returns NaNs for a sequence where every key is padded.
            safe_valid_s = valid_s.clone()
            empty_series = ~safe_valid_s.any(dim=1)
            safe_valid_s[empty_series, 0] = True
            token_s = self.series_encoder(token_s, src_key_padding_mask=~safe_valid_s)
            token_s = token_s.masked_fill(~valid_s.unsqueeze(-1), 0.0)
            token = token_s.reshape(b, s * n, self.hidden_dim)

            token_valid = (slice_mask > 0) & (series_mask.unsqueeze(-1) > 0)
            key_padding = ~token_valid.reshape(b, s * n)
            empty_study = key_padding.all(dim=1)
            if empty_study.any():
                key_padding = key_padding.clone()
                key_padding[empty_study, 0] = False

            token_planes = plane_ids.unsqueeze(-1).expand(b, s, n).reshape(b, s * n)
            plane_bias = self.label_plane_prior[:, token_planes]
            # MultiheadAttention accepts B*num_heads,Q,K for per-study routing bias.
            attention_mask = (
                plane_bias.permute(1, 0, 2)
                .repeat_interleave(num_heads, dim=0)
                .to(dtype=token.dtype)
            )
            queries = self.label_queries.unsqueeze(0).expand(b, -1, -1)
            attended, _ = self.cross_attention(
                queries,
                token,
                token,
                key_padding_mask=key_padding,
                attn_mask=attention_mask,
                need_weights=False,
            )
            attended = self.output_norm(attended + queries)
            if return_embeddings:
                return attended
            logits = (attended * self.classifier_weight.unsqueeze(0)).sum(dim=-1)
            return logits + self.classifier_bias.unsqueeze(0)

    return LabelQueryStudyModel()


def create_ordered_feature_model(
    *,
    input_dim: int,
    hidden_dim: int = 384,
    num_heads: int = 6,
    sequence_layers: int = 1,
    dropout: float = 0.1,
):
    """Shared-head control adding position and ordered series encoding only."""
    torch, nn = _torch()

    class OrderedFeatureStudyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.plane_emb = nn.Embedding(len(PLANE_TO_ID), hidden_dim)
            self.fluid_emb = nn.Embedding(2, hidden_dim)
            self.fat_emb = nn.Embedding(2, hidden_dim)
            self.position_mlp = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.series_encoder = nn.TransformerEncoder(layer, num_layers=sequence_layers)
            self.slice_pool = create_attention_pool(hidden_dim)
            self.series_pool = create_attention_pool(hidden_dim)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.classifiers = nn.ModuleList(
                [nn.Linear(hidden_dim, 1) for _ in range(NUM_LABELS)]
            )

        def forward(
            self,
            features: Any,
            plane_ids: Any,
            fluid: Any,
            fat_sup: Any,
            series_mask: Any,
            slice_mask: Any,
            slice_positions: Any,
        ) -> Any:
            b, s, n, _ = features.shape
            token = self.input_proj(features)
            meta = (
                self.plane_emb(plane_ids.long().clamp(0, len(PLANE_TO_ID) - 1))
                + self.fluid_emb(fluid.long().clamp(0, 1))
                + self.fat_emb(fat_sup.long().clamp(0, 1))
            )
            token = token + meta.unsqueeze(2) + self.position_mlp(slice_positions.unsqueeze(-1))
            token_s = token.reshape(b * s, n, hidden_dim)
            valid = slice_mask.reshape(b * s, n) > 0
            safe_valid = valid.clone()
            safe_valid[~safe_valid.any(dim=1), 0] = True
            token_s = self.series_encoder(token_s, src_key_padding_mask=~safe_valid)
            token_s = token_s.masked_fill(~valid.unsqueeze(-1), 0.0)
            series_vec = self.slice_pool(token_s, valid).reshape(b, s, hidden_dim)
            shared = self.head(self.series_pool(series_vec, series_mask))
            return torch.cat([classifier(shared) for classifier in self.classifiers], dim=-1)

    return OrderedFeatureStudyModel()
