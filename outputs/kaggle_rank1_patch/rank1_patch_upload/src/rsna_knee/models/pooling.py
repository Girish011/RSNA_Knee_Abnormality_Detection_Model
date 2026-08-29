"""Masked attention pooling over slices and series."""

from __future__ import annotations

from typing import Any


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


def create_attention_pool(dim: int):
    torch, nn, F = _torch()

    class AttentionPool(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.score = nn.Linear(dim, 1)

        def forward(self, tokens: Any, mask: Any | None = None) -> Any:
            # tokens: (B, T, D), mask: (B, T) with 1=valid
            logits = self.score(tokens).squeeze(-1)  # (B, T)
            if mask is not None:
                logits = logits.masked_fill(mask <= 0, -1e4)
            weights = F.softmax(logits, dim=-1).unsqueeze(-1)  # (B, T, 1)
            return (tokens * weights).sum(dim=1)

    return AttentionPool()
