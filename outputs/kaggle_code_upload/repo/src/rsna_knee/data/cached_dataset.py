"""Cached study dataset for training without re-decoding DICOM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rsna_knee.constants import LABEL_COLS, NUM_LABELS
from rsna_knee.data.dataset import build_label_arrays


def _torch():
    import torch
    from torch.utils.data import Dataset

    return torch, Dataset


class CachedStudyDataset:
    """Factory returning a torch Dataset over study-level .npz cache files."""

    def __new__(cls, *args, **kwargs):
        torch, Dataset = _torch()

        class _CachedStudyDataset(Dataset):
            def __init__(
                self,
                studies_df: pd.DataFrame,
                cache_dir: str | Path,
                *,
                require_cache: bool = True,
            ) -> None:
                self.studies_df = studies_df.reset_index(drop=True)
                self.cache_dir = Path(cache_dir)
                self.require_cache = require_cache
                if require_cache:
                    missing = []
                    for uid in self.studies_df["StudyInstanceUID"].astype(str):
                        if not (self.cache_dir / f"{uid}.npz").exists():
                            missing.append(uid)
                            if len(missing) > 5:
                                break
                    if missing:
                        raise FileNotFoundError(
                            f"Missing cache for e.g. {missing[:3]} under {self.cache_dir}"
                        )

            def __len__(self) -> int:
                return len(self.studies_df)

            def __getitem__(self, idx: int) -> dict[str, Any]:
                row = self.studies_df.iloc[idx]
                study_uid = str(row["StudyInstanceUID"])
                path = self.cache_dir / f"{study_uid}.npz"
                data = np.load(path, allow_pickle=True)
                images_u8 = data["images"]  # (S,N,H,W)
                images = images_u8.astype(np.float32) / 255.0
                # (S,N,3,H,W) RGB-replicated grayscale for DINOv2
                images = np.stack([images, images, images], axis=2)
                labels, label_mask, label_conf = build_label_arrays(row)
                return {
                    "study_uid": study_uid,
                    "images": torch.from_numpy(images),
                    "plane_ids": torch.from_numpy(data["plane_ids"].astype(np.int64)),
                    "fluid": torch.from_numpy(data["fluid"].astype(np.float32)),
                    "fat_sup": torch.from_numpy(data["fat_sup"].astype(np.float32)),
                    "series_mask": torch.from_numpy(data["series_mask"].astype(np.float32)),
                    "slice_mask": torch.from_numpy(data["slice_mask"].astype(np.float32)),
                    "labels": torch.from_numpy(labels),
                    "label_mask": torch.from_numpy(label_mask),
                    "label_confidence": torch.from_numpy(label_conf),
                }

        return _CachedStudyDataset(*args, **kwargs)


def merge_weak_labels(train_df: pd.DataFrame, weak_csv: str | Path | None) -> pd.DataFrame:
    """Replace NaN expert labels with weak labels when provided."""
    if weak_csv is None:
        return train_df
    weak_csv = Path(weak_csv)
    if not weak_csv.exists():
        return train_df
    weak = pd.read_csv(weak_csv)
    out = train_df.copy()
    w = weak.set_index("StudyInstanceUID")
    for idx, row in out.iterrows():
        uid = str(row["StudyInstanceUID"])
        if uid not in w.index:
            continue
        wr = w.loc[uid]
        for c in LABEL_COLS:
            if pd.isna(row.get(c)) and c in wr.index and pd.notna(wr[c]):
                out.at[idx, c] = wr[c]
                conf_col = f"{c}__conf"
                if conf_col in wr.index:
                    out.at[idx, conf_col] = wr[conf_col]
    # Ensure conf columns exist
    for c in LABEL_COLS:
        conf_col = f"{c}__conf"
        if conf_col not in out.columns:
            out[conf_col] = 1.0
        out[conf_col] = out[conf_col].fillna(1.0)
    return out


def attach_folds(train_df: pd.DataFrame, folds_csv: str | Path) -> pd.DataFrame:
    folds = pd.read_csv(folds_csv)[["StudyInstanceUID", "fold"]]
    return train_df.merge(folds, on="StudyInstanceUID", how="left")
