"""Label-aware plane routing priors for multi-series pooling."""

from __future__ import annotations

import numpy as np

from rsna_knee.constants import LABEL_COLS, LABEL_PLANE_PRIOR, NUM_LABELS, PLANE_TO_ID


def plane_prior_bias_matrix(
    *,
    prior_bonus: float = 1.5,
    non_prior_penalty: float = -0.75,
    unknown_bonus: float = 0.0,
) -> np.ndarray:
    """Build (NUM_LABELS, num_planes) additive logits bias for series attention.

    Preferred planes from ``LABEL_PLANE_PRIOR`` receive ``prior_bonus``; other
    known planes receive ``non_prior_penalty``. Unknown plane id gets
    ``unknown_bonus``.
    """
    n_planes = max(PLANE_TO_ID.values()) + 1
    bias = np.full((NUM_LABELS, n_planes), non_prior_penalty, dtype=np.float32)
    bias[:, PLANE_TO_ID["Unknown"]] = unknown_bonus
    for li, label in enumerate(LABEL_COLS):
        for plane in LABEL_PLANE_PRIOR.get(label, []):
            pid = PLANE_TO_ID.get(plane)
            if pid is not None:
                bias[li, pid] = prior_bonus
    return bias
