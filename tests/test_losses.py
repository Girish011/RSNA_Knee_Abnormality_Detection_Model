"""Robust/noisy-teacher loss tests and a CPU end-to-end model wiring check.

torch is optional in the local dev environment, so the torch-dependent tests are
skipped when it is absent (they run on Kaggle and anywhere torch is installed).
``pos_weight_from_prevalence`` is numpy-only and always runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.training.loss import pos_weight_from_prevalence

torch = pytest.importorskip("torch")

from rsna_knee.training.loss import (
    masked_bce_with_logits,
    masked_multilabel_loss,
)


def test_pos_weight_from_prevalence_rare_gets_more_weight():
    # floor=0 exposes the raw monotonic (1-p)/p relationship: rarer → larger weight.
    w = pos_weight_from_prevalence([0.5, 0.05, 0.9], cap=10.0, floor=0.0)
    assert w[1] > w[0] > w[2]  # p=0.05 rarest → largest, p=0.9 most common → smallest
    assert w[1] == pytest.approx(10.0)  # (1-0.05)/0.05 = 19 capped to 10
    # Default floor=1.0 keeps positive weights from dropping below 1.
    w_floored = pos_weight_from_prevalence([0.9], cap=10.0, floor=1.0)
    assert w_floored[0] == pytest.approx(1.0)


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(4, 12, generator=g, requires_grad=True)
    targets = (torch.rand(4, 12, generator=g) > 0.5).float()
    mask = (torch.rand(4, 12, generator=g) > 0.2).float()
    conf = torch.rand(4, 12, generator=g)
    return logits, targets, mask, conf


def test_bce_mode_matches_legacy_masked_bce():
    logits, targets, mask, conf = _batch()
    legacy = masked_bce_with_logits(logits, targets, mask, conf, pos_weight=2.0)
    new = masked_multilabel_loss(
        logits, targets, mask, conf, mode="bce", pos_weight=2.0, label_smoothing=0.0
    )
    assert torch.allclose(legacy, new, atol=1e-6)


def test_masking_zeroes_unsupervised_entries():
    logits, targets, _, _ = _batch(1)
    full = torch.ones_like(targets)
    none = torch.zeros_like(targets)
    loss_none = masked_multilabel_loss(logits, targets, none)
    assert float(loss_none.detach()) == pytest.approx(0.0)
    loss_full = masked_multilabel_loss(logits, targets, full)
    assert float(loss_full.detach()) > 0.0


def test_robust_modes_are_finite_and_differentiable():
    for mode in ("bce", "gce", "sce"):
        logits, targets, mask, conf = _batch(2)
        loss = masked_multilabel_loss(logits, targets, mask, conf, mode=mode)
        assert torch.isfinite(loss)
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_gce_is_more_robust_to_a_flipped_label_than_bce():
    # A single confidently-wrong ("flipped") target should perturb GCE less than BCE.
    logits = torch.zeros(1, 1)
    logits[0, 0] = 6.0  # model is very confident positive
    flipped = torch.zeros(1, 1)  # noisy teacher says negative
    mask = torch.ones(1, 1)
    bce = float(masked_multilabel_loss(logits, flipped, mask, mode="bce"))
    gce = float(masked_multilabel_loss(logits, flipped, mask, mode="gce", gce_q=0.7))
    assert gce < bce


def test_per_label_pos_weight_vector_broadcasts():
    logits, targets, mask, conf = _batch(3)
    pw = np.linspace(1.0, 3.0, 12)
    loss = masked_multilabel_loss(logits, targets, mask, conf, pos_weight=pw)
    assert torch.isfinite(loss)


def test_end_to_end_model_step_with_robust_loss():
    """Tiny forward+backward through the real multiseries model on CPU.

    Uses the identity-encoder fallback (pretrained=False) so no weights/network are
    needed, proving the robust loss wires into the actual model graph.
    """
    from rsna_knee.models.multiseries import create_multiseries_model

    model = create_multiseries_model("dinov2_vits14", pretrained=False, freeze_backbone=False)
    b, s, n, h = 2, 2, 3, 8
    g = torch.Generator().manual_seed(7)
    images = torch.rand(b, s, n, 3, h, h, generator=g)
    plane_ids = torch.zeros(b, s, dtype=torch.long)
    fluid = torch.zeros(b, s)
    fat = torch.zeros(b, s)
    series_mask = torch.ones(b, s)
    slice_mask = torch.ones(b, s, n)
    labels = (torch.rand(b, 12, generator=g) > 0.5).float()
    label_mask = torch.ones(b, 12)
    conf = torch.ones(b, 12)

    logits = model(images, plane_ids, fluid, fat, series_mask, slice_mask)
    assert logits.shape == (b, 12)
    loss = masked_multilabel_loss(logits, labels, label_mask, conf, mode="gce")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(gr).all() for gr in grads)
