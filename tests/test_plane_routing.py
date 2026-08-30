"""Tests for label plane routing and ACL sagittal slice bias."""

from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.data.dicom import sample_slice_indices
from rsna_knee.models.plane_routing import plane_prior_bias_matrix


def test_sagittal_acl_bias_tighter_than_default():
    n = 100
    default = sample_slice_indices(n, 12, center_bias=True, sagittal_acl_bias=False)
    acl = sample_slice_indices(n, 12, center_bias=True, sagittal_acl_bias=True)
    assert min(acl) > min(default)
    assert max(acl) < max(default)


def test_plane_prior_bias_favors_acl_sagittal():
    bias = plane_prior_bias_matrix()
    from rsna_knee.constants import LABEL_COLS, PLANE_TO_ID

    acl_i = LABEL_COLS.index("ACL")
    sag = PLANE_TO_ID["Sagittal"]
    ax = PLANE_TO_ID["Axial"]
    assert bias[acl_i, sag] > bias[acl_i, ax]


def test_plane_routing_model_forward_cpu():
    torch = pytest.importorskip("torch")
    from rsna_knee.models.multiseries import create_multiseries_model

    model = create_multiseries_model("dinov2_vits14", pretrained=False, label_plane_routing=True)
    b, s, n, h, w = 1, 3, 4, 224, 224
    images = torch.randn(b, s, n, 3, h, w)
    plane_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    fluid = torch.ones(b, s)
    fat = torch.zeros(b, s)
    series_mask = torch.tensor([[1.0, 1.0, 1.0]])
    slice_mask = torch.ones(b, s, n)
    logits = model(images, plane_ids, fluid, fat, series_mask, slice_mask)
    assert logits.shape == (1, 12)
