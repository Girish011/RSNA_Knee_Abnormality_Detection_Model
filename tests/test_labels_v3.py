"""Tests for v3 zero-shot label ensembling and the pseudo-label merge layering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsna_knee.constants import LABEL_COLS
from rsna_knee.text.zeroshot_labels import (
    HYPOTHESIS_VARIANTS,
    aggregate_variant_scores,
    merge_pseudo_labels,
    scores_to_label,
)


def test_variant_hypotheses_cover_all_labels():
    assert list(HYPOTHESIS_VARIANTS) == LABEL_COLS
    for label, variants in HYPOTHESIS_VARIANTS.items():
        assert len(variants) >= 2, label


def test_aggregate_variant_scores_methods():
    assert aggregate_variant_scores([0.2, 0.4, 0.6]) == pytest.approx(0.4)
    assert aggregate_variant_scores([0.2, 0.9], method="max") == 0.9
    assert aggregate_variant_scores([0.2, 0.9], method="min") == 0.2
    assert aggregate_variant_scores([]) == 0.0


def test_scores_to_label_abstains_in_the_middle():
    assert scores_to_label(0.9)[0] == 1.0
    assert scores_to_label(0.1)[0] == 0.0
    assert scores_to_label(0.5)[0] is None


def _empty_zs(uids):
    df = pd.DataFrame({"StudyInstanceUID": uids})
    for c in LABEL_COLS:
        df[c] = np.nan
        df[f"{c}__conf"] = np.nan
    return df


def test_merge_prefers_nli_then_v1_fallback():
    zs = _empty_zs(["s1"])
    zs.loc[0, "ACL"] = 1.0  # NLI decided ACL
    zs.loc[0, "ACL__conf"] = 0.85

    v1 = pd.DataFrame({"StudyInstanceUID": ["s1"]})
    for c in LABEL_COLS:
        v1[c] = np.nan
        v1[f"{c}__conf"] = np.nan
    v1.loc[0, "ACL"] = 0.0  # should NOT override NLI
    v1.loc[0, "ACL__conf"] = 0.9
    v1.loc[0, "Effusion"] = 1.0  # NLI abstained → v1 fills
    v1.loc[0, "Effusion__conf"] = 0.7
    v1.loc[0, "MCL"] = 1.0  # low conf → dropped
    v1.loc[0, "MCL__conf"] = 0.3

    out = merge_pseudo_labels(zs, v1=v1, min_v1_conf=0.5)
    assert out.loc[0, "ACL"] == 1.0  # NLI kept
    assert out.loc[0, "Effusion"] == 1.0  # v1 fallback applied
    assert pd.isna(out.loc[0, "MCL"])  # low-confidence v1 not used


def test_merge_expert_override_wins():
    zs = _empty_zs(["s1"])
    zs.loc[0, "ACL"] = 1.0
    zs.loc[0, "ACL__conf"] = 0.9

    expert = pd.DataFrame({"StudyInstanceUID": ["s1"]})
    for c in LABEL_COLS:
        expert[c] = 0.0  # fully labeled expert study, all negative
    out = merge_pseudo_labels(zs, expert=expert)
    assert out.loc[0, "ACL"] == 0.0  # expert overrides NLI
    assert out.loc[0, "ACL__conf"] == 1.0
