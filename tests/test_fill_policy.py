"""Tests for the per-label LLM-fill reliability policy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rsna_knee.text.fill_policy import drop_fills, unreliable_fill_labels


def test_unreliable_labels_uses_precision_threshold():
    audit = pd.DataFrame(
        {
            "label": ["ACL", "MCL", "Effusion", "Contusion"],
            "positive_precision": [0.75, 0.231, np.nan, 0.59],
        }
    )
    # Default 0.5 → only MCL (0.231). Contusion 0.59 and NaN stay.
    assert unreliable_fill_labels(audit) == ["MCL"]
    # Stricter bar pulls in Contusion too.
    assert unreliable_fill_labels(audit, min_precision=0.6) == ["MCL", "Contusion"]


def test_drop_fills_blanks_value_and_confidence():
    fills = pd.DataFrame(
        {
            "StudyInstanceUID": ["s1", "s2"],
            "MCL": [1.0, 0.0],
            "MCL__conf": [0.96, 0.99],
            "ACL": [1.0, np.nan],
            "ACL__conf": [0.97, np.nan],
        }
    )
    out = drop_fills(fills, ["MCL"])
    assert out["MCL"].isna().all()
    assert out["MCL__conf"].isna().all()
    assert out["ACL"].tolist()[:1] == [1.0]  # untouched
