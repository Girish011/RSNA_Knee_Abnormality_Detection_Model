"""Tests for ordered cache metadata and label-query aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from rsna_knee.data.dicom import make_adjacent_slice_triplets, normalized_slice_positions


def test_normalized_slice_positions_preserve_source_location():
    positions = normalized_slice_positions([2, 5, 8], 11)
    np.testing.assert_allclose(positions, [0.2, 0.5, 0.8])


def test_adjacent_slice_triplets_clamp_volume_edges():
    volume = np.arange(5, dtype=np.float32)[:, None, None]
    triplets = make_adjacent_slice_triplets(volume, [0, 2, 4])
    assert triplets.shape == (3, 3, 1, 1)
    np.testing.assert_array_equal(triplets[:, :, 0, 0], [[0, 0, 1], [1, 2, 3], [3, 4, 4]])


def test_label_query_feature_forward_shape_and_empty_series():
    torch = pytest.importorskip("torch")
    from rsna_knee.models.label_query import create_label_query_model

    model = create_label_query_model(
        input_dim=16,
        hidden_dim=24,
        num_heads=4,
        sequence_layers=1,
        dropout=0.0,
    )
    features = torch.randn(2, 3, 4, 16)
    plane_ids = torch.tensor([[0, 1, 2], [0, 1, 3]])
    fluid = torch.zeros(2, 3)
    fat = torch.zeros(2, 3)
    series_mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.float32)
    slice_mask = series_mask.unsqueeze(-1).expand(-1, -1, 4).clone()
    positions = torch.linspace(0, 1, 4).view(1, 1, 4).expand(2, 3, -1)
    logits = model(
        features,
        plane_ids,
        fluid,
        fat,
        series_mask,
        slice_mask,
        positions,
    )
    assert logits.shape == (2, 12)
    assert torch.isfinite(logits).all()


def test_feature_baseline_return_embeddings():
    torch = pytest.importorskip("torch")
    from rsna_knee.models.multiseries import create_feature_multiseries_model

    model = create_feature_multiseries_model(16, hidden_dim=16, dropout=0.0)
    features = torch.randn(2, 3, 4, 16)
    plane_ids = torch.tensor([[0, 1, 2], [0, 1, 3]])
    fluid = torch.zeros(2, 3)
    fat = torch.zeros(2, 3)
    series_mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.float32)
    slice_mask = series_mask.unsqueeze(-1).expand(-1, -1, 4).clone()
    shared = model(
        features,
        plane_ids,
        fluid,
        fat,
        series_mask,
        slice_mask,
        return_embeddings=True,
    )
    logits = model(features, plane_ids, fluid, fat, series_mask, slice_mask)
    assert shared.shape == (2, 16)
    assert logits.shape == (2, 12)
    assert torch.isfinite(shared).all()
