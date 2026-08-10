"""PyTorch study dataset (optional torch dependency)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rsna_knee.constants import LABEL_COLS, NUM_LABELS
from rsna_knee.data.dicom import gray_to_rgb, load_series_volume, sample_slice_indices
from rsna_knee.data.series import SeriesChoice, rank_and_select_series


def _maybe_torch():
    try:
        import torch
        from torch.utils.data import Dataset

        return torch, Dataset
    except ImportError as e:  # pragma: no cover
        raise ImportError("torch is required for StudyDataset") from e


def build_label_arrays(row: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, mask, confidence for one study row."""
    labels = np.zeros(NUM_LABELS, dtype=np.float32)
    mask = np.zeros(NUM_LABELS, dtype=np.float32)
    conf = np.ones(NUM_LABELS, dtype=np.float32)
    for i, col in enumerate(LABEL_COLS):
        if col not in row.index:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        labels[i] = float(val)
        mask[i] = 1.0
        conf_col = f"{col}__conf"
        if conf_col in row.index and not pd.isna(row[conf_col]):
            conf[i] = float(row[conf_col])
    return labels, mask, conf


class StudyDataset:
    """Factory that returns a torch Dataset when torch is installed."""

    def __new__(cls, *args, **kwargs):
        torch, Dataset = _maybe_torch()

        class _StudyDataset(Dataset):
            def __init__(
                self,
                studies_df: pd.DataFrame,
                series_df: pd.DataFrame,
                series_root: str | Path,
                *,
                max_series: int = 4,
                n_slices: int = 12,
                image_size: int = 224,
                train: bool = True,
            ) -> None:
                self.studies_df = studies_df.reset_index(drop=True)
                self.series_df = series_df
                self.series_root = Path(series_root)
                self.max_series = max_series
                self.n_slices = n_slices
                self.image_size = image_size
                self.train = train

            def __len__(self) -> int:
                return len(self.studies_df)

            def __getitem__(self, idx: int) -> dict[str, Any]:
                row = self.studies_df.iloc[idx]
                study_uid = str(row["StudyInstanceUID"])
                choices = rank_and_select_series(
                    self.series_df, study_uid, max_series=self.max_series
                )
                images, plane_ids, fluid, fat, series_mask, slice_mask = self._load_choices(
                    study_uid, choices
                )
                labels, label_mask, label_conf = build_label_arrays(row)
                return {
                    "study_uid": study_uid,
                    "images": torch.from_numpy(images),  # (S,N,3,H,W)
                    "plane_ids": torch.from_numpy(plane_ids),
                    "fluid": torch.from_numpy(fluid),
                    "fat_sup": torch.from_numpy(fat),
                    "series_mask": torch.from_numpy(series_mask),
                    "slice_mask": torch.from_numpy(slice_mask),
                    "labels": torch.from_numpy(labels),
                    "label_mask": torch.from_numpy(label_mask),
                    "label_confidence": torch.from_numpy(label_conf),
                }

            def _load_choices(
                self, study_uid: str, choices: list[SeriesChoice]
            ) -> tuple[np.ndarray, ...]:
                s, n, h = self.max_series, self.n_slices, self.image_size
                images = np.zeros((s, n, 3, h, h), dtype=np.float32)
                plane_ids = np.zeros((s,), dtype=np.int64)
                fluid = np.zeros((s,), dtype=np.float32)
                fat = np.zeros((s,), dtype=np.float32)
                series_mask = np.zeros((s,), dtype=np.float32)
                slice_mask = np.zeros((s, n), dtype=np.float32)

                for i, ch in enumerate(choices[:s]):
                    series_dir = self.series_root / study_uid / ch.series_uid
                    vol = load_series_volume(series_dir, target_size=(h, h))
                    idxs = sample_slice_indices(len(vol), n, center_bias=True)
                    if not idxs:
                        continue
                    rgb = gray_to_rgb(vol[idxs])  # (n',H,W,3)
                    # CHW
                    rgb = np.transpose(rgb, (0, 3, 1, 2))
                    n_keep = rgb.shape[0]
                    images[i, :n_keep] = rgb
                    slice_mask[i, :n_keep] = 1.0
                    plane_ids[i] = ch.plane_id
                    fluid[i] = float(ch.fluid_sensitive)
                    fat[i] = float(ch.fat_suppression)
                    series_mask[i] = 1.0
                return images, plane_ids, fluid, fat, series_mask, slice_mask

        return _StudyDataset(*args, **kwargs)


def collate_studies(batch: list[dict[str, Any]]) -> dict[str, Any]:
    torch, _ = _maybe_torch()
    out: dict[str, Any] = {"study_uid": [b["study_uid"] for b in batch]}
    for key in (
        "images",
        "plane_ids",
        "fluid",
        "fat_sup",
        "series_mask",
        "slice_mask",
        "labels",
        "label_mask",
        "label_confidence",
    ):
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out
