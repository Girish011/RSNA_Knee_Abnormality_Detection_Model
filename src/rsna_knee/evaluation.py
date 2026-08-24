"""Multi-fold OOF evaluation with bootstrap CIs and a pre-registered decision rule.

Lessons from the public leaderboard (a top team, ~0.937): the LB is noise-limited
in the third decimal, and single-fold eyeballing repeatedly produced false "wins".
The discipline that mattered was reading the *full* OOF across folds and deciding
against a *pre-registered* margin rather than chasing tiny deltas.

This module keeps that honest:
- ``load_oof`` / ``stack_oof`` assemble full 5-fold OOF predictions.
- ``oof_macro_auc`` scores them against a (possibly partly-unlabeled) target frame.
- ``bootstrap_macro_auc`` measures the sampling noise directly.
- ``decide`` applies a pre-registered keep/kill rule (margin + optional CI test).

Everything here is numpy/pandas/sklearn only (no torch), so it runs anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.constants import LABEL_COLS
from rsna_knee.metrics import _safe_auc, macro_auc, per_label_auc

ID_COL = "StudyInstanceUID"


def load_oof(paths: list[str | Path]) -> pd.DataFrame:
    """Concatenate per-fold OOF CSVs (ID + 12 label probability columns)."""
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        missing = [c for c in [ID_COL, *LABEL_COLS] if c not in df.columns]
        if missing:
            raise ValueError(f"{p}: OOF missing columns {missing}")
        frames.append(df[[ID_COL, *LABEL_COLS]])
    out = pd.concat(frames, ignore_index=True)
    dupes = out[ID_COL].duplicated().sum()
    if dupes:
        raise ValueError(f"{dupes} duplicate studies across OOF folds (overlapping val sets?)")
    return out


def align_targets(
    oof: pd.DataFrame,
    targets: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Join OOF probabilities to a target frame on StudyInstanceUID.

    Returns ``(y_true, y_score, mask)`` as (N, 12) arrays. ``mask`` is 1 where the
    target label is present (not NaN); ``y_true`` is NaN elsewhere so metrics ignore it.
    """
    t = targets[[ID_COL, *[c for c in LABEL_COLS if c in targets.columns]]].copy()
    merged = oof.merge(t, on=ID_COL, how="inner", suffixes=("_pred", "_true"))
    n = len(merged)
    y_true = np.full((n, len(LABEL_COLS)), np.nan, dtype=np.float64)
    y_score = np.zeros((n, len(LABEL_COLS)), dtype=np.float64)
    mask = np.zeros((n, len(LABEL_COLS)), dtype=np.float64)
    for i, c in enumerate(LABEL_COLS):
        y_score[:, i] = pd.to_numeric(merged[f"{c}_pred"], errors="coerce").to_numpy()
        true_col = f"{c}_true" if f"{c}_true" in merged.columns else None
        if true_col is None:
            continue
        col = pd.to_numeric(merged[true_col], errors="coerce")
        present = col.notna().to_numpy()
        y_true[present, i] = col[present].to_numpy()
        mask[present, i] = 1.0
    return y_true, y_score, mask


def oof_macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, object]:
    """Full-OOF macro AUC plus per-label AUC and supervised-count diagnostics."""
    per = per_label_auc(y_true, y_score)
    counts = {
        c: int(np.isfinite(y_true[:, i]).sum()) for i, c in enumerate(LABEL_COLS)
    }
    return {
        "macro_auc": macro_auc(y_true, y_score),
        "per_label_auc": per,
        "n_supervised": counts,
    }


def bootstrap_macro_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Study-level bootstrap of macro AUC → (mean, lo, hi, std).

    Resamples studies (rows) with replacement so the interval reflects the noise
    that actually moves the leaderboard. Labels that become single-class within a
    resample are skipped by ``_safe_auc`` and dropped from that macro average.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n = y_true.shape[0]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        per = [
            _safe_auc(yt[:, i], ys[:, i]) for i in range(len(LABEL_COLS))
        ]
        finite = [v for v in per if np.isfinite(v)]
        if finite:
            vals.append(float(np.mean(finite)))
    if not vals:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "std": float("nan")}
    arr = np.asarray(vals)
    return {
        "mean": float(arr.mean()),
        "lo": float(np.quantile(arr, alpha / 2)),
        "hi": float(np.quantile(arr, 1 - alpha / 2)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


@dataclass
class Decision:
    verdict: str  # "keep" | "kill" | "inconclusive"
    delta: float
    margin: float
    candidate_macro: float
    baseline_macro: float
    rationale: str


def decide(
    candidate_macro: float,
    baseline_macro: float,
    *,
    margin: float = 0.005,
    candidate_ci: tuple[float, float] | None = None,
    baseline_ci: tuple[float, float] | None = None,
) -> Decision:
    """Pre-registered keep/kill rule.

    Keep the candidate only if it beats the baseline by at least ``margin`` AND,
    when bootstrap CIs are supplied, the candidate's lower bound clears the
    baseline's mean (guards against noise-driven "wins"). Otherwise kill or, when
    the improvement is real-sized but the CIs still overlap, report inconclusive.
    """
    delta = float(candidate_macro) - float(baseline_macro)
    if delta < margin:
        return Decision(
            "kill", delta, margin, candidate_macro, baseline_macro,
            f"delta {delta:+.4f} < margin {margin:.4f}",
        )
    if (
        candidate_ci is not None
        and baseline_ci is not None
        and candidate_ci[0] <= baseline_macro
    ):
        return Decision(
            "inconclusive", delta, margin, candidate_macro, baseline_macro,
            f"delta {delta:+.4f} >= margin but candidate CI lo "
            f"{candidate_ci[0]:.4f} <= baseline mean {baseline_macro:.4f}",
        )
    return Decision(
        "keep", delta, margin, candidate_macro, baseline_macro,
        f"delta {delta:+.4f} >= margin {margin:.4f}"
        + ("" if candidate_ci is None else " and candidate CI clears baseline"),
    )


def evaluate_oof(
    oof: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, object]:
    """Convenience: align, score, and bootstrap in one call."""
    y_true, y_score, _ = align_targets(oof, targets)
    summary = oof_macro_auc(y_true, y_score)
    summary["bootstrap"] = bootstrap_macro_auc(y_true, y_score, n_boot=n_boot, seed=seed)
    summary["n_studies"] = int(y_true.shape[0])
    return summary
