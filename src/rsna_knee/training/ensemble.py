"""OOF blending utilities."""

from __future__ import annotations

import numpy as np

from rsna_knee.constants import NUM_LABELS
from rsna_knee.metrics import macro_auc


def per_label_blend_weights(
    oof_preds: list[np.ndarray],
    y_true: np.ndarray,
    label_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Grid-free uniform start; refine by leaving one model out score later.

    Returns weights shaped (M, 12) normalized per label.
    """
    m = len(oof_preds)
    weights = np.ones((m, NUM_LABELS), dtype=np.float64) / m
    # Simple: weight models by their single-model macro contribution per label.
    for j in range(NUM_LABELS):
        scores = []
        for pred in oof_preds:
            yt = y_true[:, j]
            yp = pred[:, j]
            mask = np.isfinite(yt)
            if label_mask is not None:
                mask = mask & (label_mask[:, j] > 0)
            if mask.sum() < 2 or np.unique(yt[mask]).size < 2:
                scores.append(0.5)
            else:
                from sklearn.metrics import roc_auc_score

                scores.append(float(roc_auc_score(yt[mask], yp[mask])))
        arr = np.asarray(scores, dtype=np.float64)
        arr = np.clip(arr, 1e-3, None)
        weights[:, j] = arr / arr.sum()
    return weights


def blend_predictions(preds: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Weighted blend. preds: list of (N,12), weights: (M,12)."""
    stacked = np.stack(preds, axis=0)  # (M,N,12)
    w = weights[:, None, :]  # (M,1,12)
    return (stacked * w).sum(axis=0)


def evaluate_blend(
    preds: list[np.ndarray],
    y_true: np.ndarray,
    label_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    weights = per_label_blend_weights(preds, y_true, label_mask)
    blended = blend_predictions(preds, weights)
    score = macro_auc(y_true, blended, label_mask)
    return blended, score, weights
