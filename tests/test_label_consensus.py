"""Tests for v7∩Qwen-style label consensus helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from rsna_knee.text.label_consensus import (
    agreement_stats,
    intersect_labels,
    overlay_consensus,
)


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_intersect_keeps_only_agreement():
    left = _frame(
        [
            {"StudyInstanceUID": "s1", "ACL": 1.0, "ACL__conf": 0.9, "MCL": 0.0, "MCL__conf": 0.8},
            {"StudyInstanceUID": "s2", "ACL": 1.0, "ACL__conf": 0.7, "MCL": 1.0, "MCL__conf": 0.6},
            {"StudyInstanceUID": "s3", "ACL": 0.0, "ACL__conf": 0.8, "MCL": np.nan, "MCL__conf": 0.0},
        ]
    )
    right = _frame(
        [
            {"StudyInstanceUID": "s1", "ACL": 1.0, "ACL__conf": 0.95, "MCL": 1.0, "MCL__conf": 0.9},
            {"StudyInstanceUID": "s2", "ACL": 0.0, "ACL__conf": 0.7, "MCL": 1.0, "MCL__conf": 0.5},
            {"StudyInstanceUID": "s3", "ACL": 0.0, "ACL__conf": 0.6, "MCL": 0.0, "MCL__conf": 0.7},
        ]
    )
    out = intersect_labels(left, right, labels=["ACL", "MCL"])
    # s1 ACL agree=1; MCL disagree → NaN. Conf = min.
    s1 = out.set_index("StudyInstanceUID").loc["s1"]
    assert float(s1["ACL"]) == 1.0
    assert float(s1["ACL__conf"]) == 0.9
    assert pd.isna(s1["MCL"])
    # s2 ACL disagree; MCL agree=1
    s2 = out.set_index("StudyInstanceUID").loc["s2"]
    assert pd.isna(s2["ACL"])
    assert float(s2["MCL"]) == 1.0
    assert float(s2["MCL__conf"]) == 0.5
    # s3 ACL agree=0; MCL one-sided → NaN
    s3 = out.set_index("StudyInstanceUID").loc["s3"]
    assert float(s3["ACL"]) == 0.0
    assert pd.isna(s3["MCL"])


def test_intersect_outer_keeps_unilateral_study_as_all_nan():
    left = _frame([{"StudyInstanceUID": "only_l", "ACL": 1.0}])
    right = _frame([{"StudyInstanceUID": "only_r", "ACL": 0.0}])
    out = intersect_labels(left, right, labels=["ACL"], how="outer")
    assert set(out["StudyInstanceUID"]) == {"only_l", "only_r"}
    assert out["ACL"].isna().all()


def test_agreement_stats_rates():
    left = _frame(
        [
            {"StudyInstanceUID": "a", "ACL": 1.0, "MCL": 0.0},
            {"StudyInstanceUID": "b", "ACL": 1.0, "MCL": np.nan},
        ]
    )
    right = _frame(
        [
            {"StudyInstanceUID": "a", "ACL": 1.0, "MCL": 1.0},
            {"StudyInstanceUID": "b", "ACL": 0.0, "MCL": 0.0},
        ]
    )
    stats = agreement_stats(left, right, labels=["ACL", "MCL"]).set_index("label")
    assert stats.loc["ACL", "n_both_commit"] == 2
    assert stats.loc["ACL", "n_agree"] == 1
    assert abs(float(stats.loc["ACL", "agree_rate"]) - 0.5) < 1e-9
    assert stats.loc["MCL", "n_both_commit"] == 1
    assert stats.loc["MCL", "n_agree"] == 0
    assert stats.loc["MCL", "n_left_only"] == 0
    assert stats.loc["MCL", "n_right_only"] == 1


def test_overlay_consensus_overwrite_and_gapfill():
    base = _frame(
        [
            {"StudyInstanceUID": "s1", "ACL": 1.0, "ACL__conf": 0.5, "MCL": np.nan, "MCL__conf": np.nan},
            {"StudyInstanceUID": "s2", "ACL": 0.0, "ACL__conf": 0.5, "MCL": 1.0, "MCL__conf": 0.9},
        ]
    )
    cons = _frame(
        [
            {"StudyInstanceUID": "s1", "ACL": 0.0, "ACL__conf": 0.8, "MCL": 1.0, "MCL__conf": 0.7},
            {"StudyInstanceUID": "s2", "ACL": np.nan, "MCL": np.nan},
        ]
    )
    filled = overlay_consensus(base, cons, labels=["ACL", "MCL"], only_where_base_isna=True)
    f1 = filled.set_index("StudyInstanceUID").loc["s1"]
    assert float(f1["ACL"]) == 1.0  # base kept
    assert float(f1["MCL"]) == 1.0  # gap filled
    over = overlay_consensus(base, cons, labels=["ACL", "MCL"], only_where_base_isna=False)
    o1 = over.set_index("StudyInstanceUID").loc["s1"]
    assert float(o1["ACL"]) == 0.0  # overwritten
    assert float(o1["ACL__conf"]) == 0.8
