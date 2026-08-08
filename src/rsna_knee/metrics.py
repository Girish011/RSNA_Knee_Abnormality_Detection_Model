"""Metrics: macro AUC and per-label diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from rsna_knee.constants import LABEL_COLS, NUM_LABELS


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return AUC or NaN if a class is missing."""
    y_true = np.asarray(y_true).astype(np.float64)
    y_score = np.asarray(y_score).astype(np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def per_label_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute AUC for each of the 12 labels.

    Args:
        y_true: (N, 12) ground truth; NaN means unsupervised.
        y_score: (N, 12) predicted probabilities.
        label_mask: optional (N, 12) 1 where label is valid.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if label_mask is not None:
        mask = np.asarray(label_mask, dtype=bool)
        y_true = y_true.copy()
        y_true[~mask] = np.nan

    out: dict[str, float] = {}
    for i, name in enumerate(LABEL_COLS):
        out[name] = _safe_auc(y_true[:, i], y_score[:, i])
    return out


def macro_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_mask: np.ndarray | None = None,
) -> float:
    """Mean of per-label AUCs, ignoring labels that are NaN (undefined)."""
    scores = per_label_auc(y_true, y_score, label_mask)
    vals = [v for v in scores.values() if np.isfinite(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def summarize_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return macro AUC plus per-label breakdown."""
    per = per_label_auc(y_true, y_score, label_mask)
    return {
        "macro_auc": macro_auc(y_true, y_score, label_mask),
        "per_label_auc": per,
        "n_defined_labels": int(sum(np.isfinite(v) for v in per.values())),
        "n_labels": NUM_LABELS,
    }
