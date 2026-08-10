"""Slice sampling helpers."""

from rsna_knee.data.dicom import sample_slice_indices


def test_sample_slice_indices_center_bias():
    idx = sample_slice_indices(100, 10, center_bias=True)
    assert len(idx) == 10
    assert idx[0] >= 10
    assert idx[-1] <= 90


def test_sample_slice_indices_small_volume():
    assert sample_slice_indices(5, 10) == [0, 1, 2, 3, 4]
