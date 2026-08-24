"""Tests for multi-fold OOF evaluation, bootstrap CIs, and the decision rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsna_knee.constants import LABEL_COLS
from rsna_knee.evaluation import (
    align_targets,
    bootstrap_macro_auc,
    decide,
    load_oof,
    oof_macro_auc,
)


def _oof_frame(uids, rng):
    data = {"StudyInstanceUID": uids}
    for c in LABEL_COLS:
        data[c] = rng.random(len(uids))
    return pd.DataFrame(data)


def test_load_oof_concats_and_rejects_dupes(tmp_path):
    rng = np.random.default_rng(0)
    f0 = _oof_frame(["a", "b"], rng)
    f1 = _oof_frame(["c", "d"], rng)
    p0, p1 = tmp_path / "f0.csv", tmp_path / "f1.csv"
    f0.to_csv(p0, index=False)
    f1.to_csv(p1, index=False)
    out = load_oof([p0, p1])
    assert len(out) == 4
    assert list(out.columns) == ["StudyInstanceUID", *LABEL_COLS]

    dup = tmp_path / "dup.csv"
    f0.to_csv(dup, index=False)
    with pytest.raises(ValueError):
        load_oof([p0, dup])  # overlapping "a","b"


def test_align_targets_masks_missing_labels():
    uids = ["a", "b", "c"]
    rng = np.random.default_rng(1)
    oof = _oof_frame(uids, rng)
    targets = pd.DataFrame({"StudyInstanceUID": uids})
    for c in LABEL_COLS:
        targets[c] = [1.0, np.nan, 0.0]
    y_true, _y_score, mask = align_targets(oof, targets)
    assert y_true.shape == (3, len(LABEL_COLS))
    assert mask[:, 0].tolist() == [1.0, 0.0, 1.0]  # middle study unlabeled
    assert np.isnan(y_true[1, 0])


def test_oof_macro_auc_perfect():
    uids = [f"u{i}" for i in range(20)]
    y = np.random.RandomState(0).randint(0, 2, size=(20, len(LABEL_COLS))).astype(float)
    oof = pd.DataFrame({"StudyInstanceUID": uids})
    targets = pd.DataFrame({"StudyInstanceUID": uids})
    for i, c in enumerate(LABEL_COLS):
        oof[c] = y[:, i]  # predictions == truth → AUC 1.0 where both classes present
        targets[c] = y[:, i]
    y_true, y_score, _ = align_targets(oof, targets)
    summary = oof_macro_auc(y_true, y_score)
    assert summary["macro_auc"] == pytest.approx(1.0)


def test_bootstrap_ci_orders_and_brackets_mean():
    rng = np.random.default_rng(2)
    n = 200
    y_true = rng.integers(0, 2, size=(n, len(LABEL_COLS))).astype(float)
    y_score = y_true * 0.6 + rng.random((n, len(LABEL_COLS))) * 0.4
    b = bootstrap_macro_auc(y_true, y_score, n_boot=200, seed=3)
    assert b["lo"] <= b["mean"] <= b["hi"]
    assert 0.5 < b["mean"] <= 1.0


def test_decision_rule_keep_kill_inconclusive():
    keep = decide(0.80, 0.79, margin=0.005, candidate_ci=(0.792, 0.808), baseline_ci=(0.782, 0.798))
    assert keep.verdict == "keep"

    kill = decide(0.792, 0.790, margin=0.005)
    assert kill.verdict == "kill"  # delta below margin

    incon = decide(0.80, 0.79, margin=0.005, candidate_ci=(0.785, 0.815), baseline_ci=(0.78, 0.80))
    assert incon.verdict == "inconclusive"  # meets margin but CI overlaps baseline mean
