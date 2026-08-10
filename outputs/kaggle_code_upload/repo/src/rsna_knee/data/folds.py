"""Study-level multilabel fold creation (leakage-safe)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from rsna_knee.constants import LABEL_COLS


def _stratify_key(df: pd.DataFrame) -> np.ndarray:
    """Build a coarse stratification key from rare-ish positive patterns."""
    # Prefer rare labels for stratification signal when present.
    rare_first = ["Fracture", "Baker's", "MCL", "Synovitis", "Contusion", "ACL"]
    cols = [c for c in rare_first if c in df.columns]
    if not cols:
        return np.zeros(len(df), dtype=int)
    mat = df[cols].fillna(0).astype(int).to_numpy()
    # Hash presence pattern into a modest number of bins.
    weights = (2 ** np.arange(mat.shape[1])).astype(int)
    keys = (mat * weights).sum(axis=1)
    # Collapse rare keys so StratifiedKFold does not fail.
    vc = pd.Series(keys).value_counts()
    rare = set(vc[vc < 2].index.tolist())
    if rare:
        keys = np.array([0 if k in rare else k for k in keys], dtype=int)
    return keys


def make_study_folds(
    train_df: pd.DataFrame,
    *,
    n_folds: int = 5,
    seed: int = 42,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Assign each study a fold id in [0, n_folds).

    - Splits are study-level (one row per StudyInstanceUID).
    - If group_col is provided (site/scanner proxy), all rows in a group share a fold.
    - Uses stratified keys when possible; falls back to KFold.
    """
    df = train_df.copy()
    if "StudyInstanceUID" not in df.columns:
        raise ValueError("train_df must include StudyInstanceUID")

    # One row per study.
    study = df.drop_duplicates("StudyInstanceUID").reset_index(drop=True)

    if group_col and group_col in study.columns:
        groups = study[group_col].fillna("UNK").astype(str)
        unique_groups = groups.unique()
        # Stratify groups by majority rare-label rate if possible.
        gdf = study.copy()
        gdf["_group"] = groups
        # Assign folds on unique groups then map back.
        ug = pd.DataFrame({"_group": unique_groups})
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_map = {}
        for fold, (_, val_idx) in enumerate(kf.split(ug)):
            for gi in val_idx:
                fold_map[ug.iloc[gi]["_group"]] = fold
        study["fold"] = groups.map(fold_map).astype(int)
    else:
        y = _stratify_key(study)
        try:
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            splits = list(splitter.split(study, y))
        except ValueError:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            splits = list(splitter.split(study))
        study["fold"] = -1
        for fold, (_, val_idx) in enumerate(splits):
            study.loc[val_idx, "fold"] = fold

    out = study[["StudyInstanceUID", "fold"]].copy()
    # Carry label presence counts for diagnostics.
    for c in LABEL_COLS:
        if c in study.columns:
            out[c] = study[c].values
    return out


def save_folds(folds: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        folds.to_parquet(path, index=False)
    else:
        folds.to_csv(path, index=False)
    return path


def load_folds(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
