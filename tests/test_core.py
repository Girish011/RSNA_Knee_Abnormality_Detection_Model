"""Tests for metrics, series selection, folds, weak labels, submission schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rsna_knee.constants import LABEL_COLS, NUM_LABELS
from rsna_knee.data.folds import make_study_folds
from rsna_knee.data.series import rank_and_select_series, score_series_row
from rsna_knee.infer import validate_submission, write_submission
from rsna_knee.metrics import macro_auc, per_label_auc
from rsna_knee.text.weak_labels import extract_label_from_report


def test_macro_auc_perfect():
    y = np.array([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.float64)
    # Pad to 12 labels
    y_true = np.zeros((4, NUM_LABELS), dtype=np.float64)
    y_score = np.zeros((4, NUM_LABELS), dtype=np.float64)
    y_true[:, :2] = y
    y_score[:, :2] = y
    # Remaining labels constant → undefined AUC ignored
    score = macro_auc(y_true, y_score)
    assert score == pytest.approx(1.0)


def test_per_label_auc_keys():
    y_true = np.random.RandomState(0).randint(0, 2, size=(32, NUM_LABELS)).astype(float)
    y_score = np.random.RandomState(1).rand(32, NUM_LABELS)
    out = per_label_auc(y_true, y_score)
    assert list(out.keys()) == LABEL_COLS


def test_series_ranking_prefers_fluid_fat():
    row_good = pd.Series(
        {
            "Fluid_Sensitive": 1,
            "Fat_Suppression": 1,
            "Anatomical_Plane": "Sagittal",
        }
    )
    row_bad = pd.Series(
        {
            "Fluid_Sensitive": 0,
            "Fat_Suppression": 0,
            "Anatomical_Plane": "Unknown",
        }
    )
    assert score_series_row(row_good) > score_series_row(row_bad)


def test_rank_and_select_plane_coverage():
    rows = []
    for i, plane in enumerate(["Sagittal", "Coronal", "Axial", "Sagittal"]):
        rows.append(
            {
                "StudyInstanceUID": "S1",
                "SeriesInstanceUID": f"SER{i}",
                "Fluid_Sensitive": 1 if i < 3 else 0,
                "Fat_Suppression": 1 if i < 3 else 0,
                "Anatomical_Plane": plane,
            }
        )
    df = pd.DataFrame(rows)
    chosen = rank_and_select_series(df, "S1", max_series=3, require_plane_coverage=True)
    planes = {c.plane for c in chosen}
    assert planes >= {"Sagittal", "Coronal", "Axial"}
    assert len(chosen) == 3


def test_make_study_folds_study_level():
    rng = np.random.RandomState(0)
    n = 50
    data = {"StudyInstanceUID": [f"u{i}" for i in range(n)]}
    for c in LABEL_COLS:
        data[c] = rng.randint(0, 2, size=n)
    df = pd.DataFrame(data)
    folds = make_study_folds(df, n_folds=5, seed=42)
    assert len(folds) == n
    assert set(folds["fold"].unique()) == {0, 1, 2, 3, 4}


def test_weak_label_acl_positive_and_negation():
    pos = extract_label_from_report("There is a complete ACL tear with discontinuity.", "ACL")
    assert pos.value == 1
    assert pos.confidence >= 0.5
    neg = extract_label_from_report("No ACL tear is identified.", "ACL")
    assert neg.value == 0
    chronic = extract_label_from_report("Remote ACL tear with chronic changes.", "ACL")
    assert chronic.value == 0
    fr = extract_label_from_report("Rupture complète du LCA avec épanchement.", "ACL")
    assert fr.value == 1
    fr_eff = extract_label_from_report("Important épanchement articulaire.", "Effusion")
    assert fr_eff.value == 1


def test_write_submission_schema(tmp_path):
    uids = ["a", "b", "c"]
    probs = np.full((3, NUM_LABELS), 0.5)
    path = write_submission(uids, probs, tmp_path / "submission.csv")
    df = pd.read_csv(path)
    validate_submission(df, expected_ids=uids)
    assert list(df.columns) == ["StudyInstanceUID"] + LABEL_COLS
