"""Compare checkpoint-selection policies offline from per-epoch OOF predictions.

Motivation (2026-09-16): the greenfield recipe shipped whichever epoch peaked on the
*weak* validation ruler, which demonstrably disagrees with the gold ruler, and it stopped
while weak-val was still rising. Both choices add run-to-run variance. If a training run
saves val predictions after every epoch (``train_baseline_fold.py --save-epoch-oof``),
every selection policy can be scored afterwards from the same trained models, so comparing
policies costs no extra GPU and is paired within a single run.

A "pick" maps fold -> list of epochs whose predictions are averaged for that fold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.constants import LABEL_COLS

ID_COL = "StudyInstanceUID"


def epoch_oof_path(out_dir: Path, fold: int, epoch: int) -> Path:
    return Path(out_dir) / f"fold{fold}_ep{epoch}_oof.csv"


def assemble_oof(out_dir: Path, pick: dict[int, list[int]]) -> pd.DataFrame:
    """Build a full OOF frame, averaging the chosen epochs' predictions within each fold.

    Folds hold disjoint validation studies, so concatenating them yields one prediction
    per study. Raises if that invariant breaks (overlapping folds / duplicate rows).
    """
    if not pick:
        raise ValueError("pick is empty")
    parts = []
    for fold, epochs in sorted(pick.items()):
        if not epochs:
            raise ValueError(f"fold {fold}: no epochs selected")
        frames = []
        for e in epochs:
            p = epoch_oof_path(out_dir, fold, e)
            if not p.exists():
                raise FileNotFoundError(p)
            df = pd.read_csv(p)
            missing = [c for c in [ID_COL, *LABEL_COLS] if c not in df.columns]
            if missing:
                raise ValueError(f"{p}: missing columns {missing}")
            frames.append(df[[ID_COL, *LABEL_COLS]])
        ids = frames[0][ID_COL].tolist()
        for fr in frames[1:]:
            if fr[ID_COL].tolist() != ids:
                raise ValueError(f"fold {fold}: epoch OOF rows are not aligned by study")
        stacked = np.mean([fr[LABEL_COLS].to_numpy(dtype=float) for fr in frames], axis=0)
        out = frames[0][[ID_COL]].copy()
        for i, c in enumerate(LABEL_COLS):
            out[c] = stacked[:, i]
        parts.append(out)
    oof = pd.concat(parts, ignore_index=True)
    if not oof[ID_COL].is_unique:
        raise ValueError("duplicate studies across folds (overlapping validation sets?)")
    return oof


def best_weakval_pick(
    histories: dict[int, list[dict]],
    *,
    limit: int | None = None,
) -> dict[int, list[int]]:
    """Per fold, the single epoch with the highest weak-val macro AUC.

    ``limit`` truncates the search to the first N epochs, which is how the original
    5-epoch recipe is reproduced from a longer run.
    """
    pick = {}
    for fold, hist in histories.items():
        rows = hist[:limit] if limit is not None else hist
        if not rows:
            raise ValueError(f"fold {fold}: empty history")
        scores = [float(r["val_macro_auc"]) for r in rows]
        pick[fold] = [int(np.argmax(scores))]
    return pick


def final_epoch_pick(folds: list[int], epochs: int) -> dict[int, list[int]]:
    """Ignore weak-val entirely and take the last epoch."""
    return {f: [epochs - 1] for f in folds}


def last_k_pick(folds: list[int], epochs: int, k: int) -> dict[int, list[int]]:
    """Average the final ``k`` epochs' predictions (checkpoint averaging, prediction-side)."""
    if not 1 <= k <= epochs:
        raise ValueError(f"k={k} out of range for {epochs} epochs")
    return {f: list(range(epochs - k, epochs)) for f in folds}


def single_epoch_pick(folds: list[int], epoch: int) -> dict[int, list[int]]:
    """All folds read at the same epoch (used for the per-epoch curve)."""
    return {f: [epoch] for f in folds}
