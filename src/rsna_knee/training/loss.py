"""Masked, confidence-weighted multilabel losses for a noisy report-derived teacher.

The competition test set is graded by expert radiologists reading the images, while
our training targets are noisy report-derived weak labels. The central lever is
therefore *learning from a noisy teacher*, so this module offers robust-loss options
(label smoothing, Generalized Cross Entropy, Symmetric Cross Entropy) on top of the
plain masked BCE. All variants keep the existing masking + per-example confidence
weighting so unsupervised (label_mask==0) entries never contribute.

Defaults reproduce the original ``masked_bce_with_logits`` behaviour exactly.
"""

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
    Kept as the historical default entry point; equivalent to
    ``masked_multilabel_loss(..., mode="bce", label_smoothing=0.0)``.
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


def _smooth_targets(targets: Any, eps: float):
    """Two-sided binary label smoothing: 1 -> 1-eps, 0 -> eps."""
    if eps <= 0.0:
        return targets
    return targets * (1.0 - eps) + (1.0 - targets) * eps


def _pos_weight_map(pos_weight: Any, targets: Any):
    """Return a per-element multiplier that upweights positive targets.

    ``pos_weight`` may be a python float or a 1-D tensor/sequence of length C
    (broadcast over the label dimension, assumed to be the last axis).
    """
    import torch

    if isinstance(pos_weight, (int, float)):
        if float(pos_weight) == 1.0:
            return None
        pw = torch.as_tensor(float(pos_weight), device=targets.device, dtype=targets.dtype)
    else:
        pw = torch.as_tensor(pos_weight, device=targets.device, dtype=targets.dtype)
        # Broadcast a (C,) vector against (..., C) targets.
        shape = [1] * (targets.dim() - 1) + [pw.numel()]
        pw = pw.reshape(shape)
    return torch.where(targets > 0.5, pw, torch.ones_like(targets))


def masked_multilabel_loss(
    logits: Any,
    targets: Any,
    mask: Any,
    confidence: Any | None = None,
    *,
    mode: str = "bce",
    pos_weight: Any = 1.0,
    label_smoothing: float = 0.0,
    gce_q: float = 0.7,
    sce_alpha: float = 1.0,
    sce_beta: float = 1.0,
    rce_clip: float = -4.0,
):
    """Masked, confidence-weighted multilabel loss with robust-teacher options.

    Args:
        logits: (..., C) raw logits.
        targets: (..., C) in {0,1} (soft targets allowed).
        mask: (..., C) 1 where the label is supervised, else 0.
        confidence: optional (..., C) per-label confidence weight in [0, 1].
        mode: one of ``"bce"``, ``"gce"``, ``"sce"``.
            - ``bce``: standard binary cross entropy.
            - ``gce``: Generalized Cross Entropy (Zhang & Sabuncu 2018),
              ``(1 - p_t**q) / q`` — robust to label noise; ``q->0`` recovers CE,
              ``q=1`` behaves like MAE.
            - ``sce``: Symmetric Cross Entropy (Wang et al. 2019),
              ``alpha*CE + beta*RCE`` with a clipped reverse term.
        pos_weight: scalar or per-label (C,) positive upweighting.
        label_smoothing: two-sided smoothing epsilon applied to targets.
        gce_q: q in (0, 1] for GCE.
        sce_alpha, sce_beta, rce_clip: SCE mixing weights and log clip.

    Returns:
        Scalar loss averaged over supervised, confidence-weighted entries.
    """
    import torch
    import torch.nn.functional as F

    targets = targets.to(logits.dtype)
    smooth = _smooth_targets(targets, label_smoothing)

    if mode == "bce":
        loss = F.binary_cross_entropy_with_logits(logits, smooth, reduction="none")
    elif mode == "gce":
        # p_t is the model probability assigned to the (smoothed) target class.
        p = torch.sigmoid(logits)
        p_t = smooth * p + (1.0 - smooth) * (1.0 - p)
        p_t = p_t.clamp_min(1e-6)
        loss = (1.0 - p_t.pow(gce_q)) / gce_q
    elif mode == "sce":
        ce = F.binary_cross_entropy_with_logits(logits, smooth, reduction="none")
        p = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
        # Reverse CE: treat predictions as the reference distribution and the
        # (clipped) targets as the noisy prediction.
        t = smooth.clamp(1e-4, 1.0 - 1e-4)
        log_t = torch.log(t).clamp_min(rce_clip)
        log_1mt = torch.log(1.0 - t).clamp_min(rce_clip)
        rce = -(p * log_t + (1.0 - p) * log_1mt)
        loss = sce_alpha * ce + sce_beta * rce
    else:
        raise ValueError(f"unknown loss mode: {mode!r} (expected bce|gce|sce)")

    pw_map = _pos_weight_map(pos_weight, targets)
    if pw_map is not None:
        loss = loss * pw_map

    weight = mask.to(logits.dtype)
    if confidence is not None:
        weight = weight * confidence.to(logits.dtype)
    denom = weight.sum().clamp_min(1.0)
    return (loss * weight).sum() / denom


def pos_weight_from_prevalence(
    prevalence: Any,
    *,
    cap: float = 10.0,
    floor: float = 1.0,
):
    """Per-label positive weight ``(1 - p) / p`` from label prevalence.

    Rare labels (small ``p``) receive a larger weight, clamped to ``[floor, cap]``.
    Accepts and returns a numpy array so it can be computed offline from the
    training folds and passed into the loss.
    """
    import numpy as np

    p = np.asarray(prevalence, dtype=np.float64)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    w = (1.0 - p) / p
    return np.clip(w, floor, cap)
