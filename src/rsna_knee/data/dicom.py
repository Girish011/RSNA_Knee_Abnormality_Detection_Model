"""DICOM decode, orientation, windowing, and resize helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None  # type: ignore


def require_pydicom() -> None:
    if pydicom is None:
        raise ImportError("pydicom is required for DICOM I/O. pip install pydicom")


def list_dicom_paths(series_dir: str | Path) -> list[Path]:
    """List .dcm files under a series directory (non-recursive)."""
    series_dir = Path(series_dir)
    if not series_dir.is_dir():
        return []
    return sorted(p for p in series_dir.iterdir() if p.suffix.lower() == ".dcm")


def _instance_number(ds) -> float:
    val = getattr(ds, "InstanceNumber", None)
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def _image_position_z(ds) -> float:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is None or len(ipp) < 3:
        return 0.0
    try:
        return float(ipp[2])
    except (TypeError, ValueError):
        return 0.0


def load_series_volume(
    series_dir: str | Path,
    *,
    target_size: tuple[int, int] = (224, 224),
    percentile_window: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """Load a DICOM series as float32 volume shaped (N, H, W) in [0, 1]."""
    require_pydicom()
    paths = list_dicom_paths(series_dir)
    if not paths:
        return np.zeros((0, target_size[0], target_size[1]), dtype=np.float32)

    records: list[tuple[float, float, np.ndarray]] = []
    for path in paths:
        try:
            ds = pydicom.dcmread(str(path), force=True)
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            continue
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2:
            continue
        records.append((_image_position_z(ds), _instance_number(ds), arr))

    if not records:
        return np.zeros((0, target_size[0], target_size[1]), dtype=np.float32)

    records.sort(key=lambda t: (t[0], t[1]))
    slices = [_window_and_resize(a, target_size, percentile_window) for _, _, a in records]
    return np.stack(slices, axis=0)


def _window_and_resize(
    arr: np.ndarray,
    target_size: tuple[int, int],
    percentile_window: tuple[float, float],
) -> np.ndarray:
    lo, hi = np.percentile(arr, percentile_window)
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return _resize2d(norm, target_size)


def _resize2d(arr: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    try:
        import cv2

        return cv2.resize(arr, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image

        img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")
        img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


def sample_slice_indices(
    n_slices: int,
    n_sample: int,
    center_bias: bool = True,
    *,
    sagittal_acl_bias: bool = False,
) -> list[int]:
    """Select up to n_sample indices from a series.

    When ``sagittal_acl_bias`` is True, sample a tighter mid-stack window
    (35–65%) where ACL / meniscus signal is often concentrated on sagittal PD.
    """
    if n_slices <= 0 or n_sample <= 0:
        return []
    if n_slices <= n_sample:
        return list(range(n_slices))
    if not center_bias:
        return np.linspace(0, n_slices - 1, n_sample).round().astype(int).tolist()
    if sagittal_acl_bias:
        lo = int(0.35 * (n_slices - 1))
        hi = int(0.65 * (n_slices - 1))
    else:
        lo = int(0.15 * (n_slices - 1))
        hi = int(0.85 * (n_slices - 1))
    if hi <= lo:
        lo, hi = 0, n_slices - 1
    return np.linspace(lo, hi, n_sample).round().astype(int).tolist()


def gray_to_rgb(volume: np.ndarray) -> np.ndarray:
    """(N, H, W) -> (N, H, W, 3) float32."""
    if volume.ndim != 3:
        raise ValueError(f"Expected (N,H,W), got {volume.shape}")
    return np.stack([volume, volume, volume], axis=-1)


def iter_study_series_dirs(study_dir: str | Path) -> Iterable[Path]:
    study_dir = Path(study_dir)
    if not study_dir.is_dir():
        return []
    return sorted(p for p in study_dir.iterdir() if p.is_dir())
