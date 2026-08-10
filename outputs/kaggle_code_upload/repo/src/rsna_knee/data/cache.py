"""Cache builder helpers for resized series tensors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rsna_knee.data.dicom import load_series_volume, sample_slice_indices


def cache_key(study_uid: str, series_uid: str) -> str:
    return f"{study_uid}__{series_uid}"


def write_series_cache(
    series_dir: str | Path,
    out_path: str | Path,
    *,
    image_size: int = 224,
    n_slices: int = 12,
    center_bias: bool = True,
) -> dict:
    """Decode a series, sample slices, save .npz, return manifest entry."""
    volume = load_series_volume(series_dir, target_size=(image_size, image_size))
    idx = sample_slice_indices(len(volume), n_slices, center_bias=center_bias)
    sampled = volume[idx] if idx else volume
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, images=sampled.astype(np.float32), indices=np.asarray(idx))
    return {
        "path": str(out_path),
        "n_source_slices": int(len(volume)),
        "n_cached_slices": int(len(sampled)),
        "image_size": image_size,
    }


def write_manifest(entries: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2))
