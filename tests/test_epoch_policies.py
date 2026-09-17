"""Tests for offline checkpoint-selection policy comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsna_knee.constants import LABEL_COLS
from rsna_knee.epoch_policies import (
    assemble_oof,
    best_weakval_pick,
    final_epoch_pick,
    last_k_pick,
    single_epoch_pick,
)


def _write_epoch_oof(d, fold: int, epoch: int, uids: list[str], value: float) -> None:
    df = pd.DataFrame({"StudyInstanceUID": uids})
    for c in LABEL_COLS:
        df[c] = value
    df.to_csv(d / f"fold{fold}_ep{epoch}_oof.csv", index=False)


def test_assemble_oof_averages_selected_epochs(tmp_path):
    _write_epoch_oof(tmp_path, 0, 0, ["a", "b"], 0.0)
    _write_epoch_oof(tmp_path, 0, 1, ["a", "b"], 1.0)
    oof = assemble_oof(tmp_path, {0: [0, 1]})
    assert len(oof) == 2
    assert np.allclose(oof[LABEL_COLS].to_numpy(dtype=float), 0.5)


def test_assemble_oof_concatenates_disjoint_folds(tmp_path):
    _write_epoch_oof(tmp_path, 0, 3, ["a"], 0.2)
    _write_epoch_oof(tmp_path, 1, 3, ["b"], 0.8)
    oof = assemble_oof(tmp_path, single_epoch_pick([0, 1], 3))
    assert sorted(oof["StudyInstanceUID"]) == ["a", "b"]
    got = dict(zip(oof["StudyInstanceUID"], oof["ACL"]))
    assert got["a"] == pytest.approx(0.2)
    assert got["b"] == pytest.approx(0.8)


def test_assemble_oof_rejects_overlapping_folds(tmp_path):
    _write_epoch_oof(tmp_path, 0, 0, ["dup"], 0.1)
    _write_epoch_oof(tmp_path, 1, 0, ["dup"], 0.9)
    with pytest.raises(ValueError, match="duplicate studies"):
        assemble_oof(tmp_path, single_epoch_pick([0, 1], 0))


def test_assemble_oof_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        assemble_oof(tmp_path, {0: [7]})


def test_best_weakval_pick_and_limit():
    # Peak inside the first 5 epochs is 4; the global peak is 7.
    hist = [
        {"epoch": 0, "val_macro_auc": 0.60},
        {"epoch": 1, "val_macro_auc": 0.62},
        {"epoch": 2, "val_macro_auc": 0.61},
        {"epoch": 3, "val_macro_auc": 0.63},
        {"epoch": 4, "val_macro_auc": 0.65},
        {"epoch": 5, "val_macro_auc": 0.64},
        {"epoch": 6, "val_macro_auc": 0.66},
        {"epoch": 7, "val_macro_auc": 0.70},
        {"epoch": 8, "val_macro_auc": 0.69},
        {"epoch": 9, "val_macro_auc": 0.68},
    ]
    assert best_weakval_pick({0: hist}, limit=5) == {0: [4]}
    assert best_weakval_pick({0: hist}) == {0: [7]}


def test_final_and_last_k_picks():
    assert final_epoch_pick([0, 1], 10) == {0: [9], 1: [9]}
    assert last_k_pick([0], 10, 3) == {0: [7, 8, 9]}
    assert last_k_pick([0], 10, 1) == {0: [9]}
    with pytest.raises(ValueError):
        last_k_pick([0], 10, 0)
    with pytest.raises(ValueError):
        last_k_pick([0], 10, 11)


def test_assemble_oof_rejects_misaligned_rows(tmp_path):
    _write_epoch_oof(tmp_path, 0, 0, ["a", "b"], 0.1)
    _write_epoch_oof(tmp_path, 0, 1, ["b", "a"], 0.9)
    with pytest.raises(ValueError, match="not aligned"):
        assemble_oof(tmp_path, {0: [0, 1]})
