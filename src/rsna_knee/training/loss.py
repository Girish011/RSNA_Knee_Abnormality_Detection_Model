"""Masked confidence-weighted multilabel BCE with optional positive upweight."""

from __future__ import annotations

from typing import Any


def masked_bce_with_logits(
    logits: Any,
    targets: Any,
    mask: Any,
    confidence: Any | None = None,
    *,
    pos_weight: float = 1.0,
):
    """BCE only where mask==1.

    pos_weight > 1 upweights positive examples (helps rare labels / macro AUC).
    """
    import torch
    import torch.nn.functional as F

    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if pos_weight != 1.0:
        pw = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
        loss = loss * torch.where(targets > 0.5, pw, torch.ones_like(targets))

    weight = mask.float()
    if confidence is not None:
        weight = weight * confidence.float()
    denom = weight.sum().clamp_min(1.0)
    return (loss * weight).sum() / denom
