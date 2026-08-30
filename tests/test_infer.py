"""Tests for inference helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna_knee.constants import LABEL_COLS, SUBMISSION_ID_COL
from rsna_knee.infer import load_oof_blend_weights, write_submission


def test_load_oof_blend_weights_normalizes_per_label(tmp_path):
    train = pd.DataFrame(
        {
            SUBMISSION_ID_COL: ["s1", "s2", "s3", "s4"],
            **{c: [0, 1, 0, 1] for c in LABEL_COLS},
        }
    )
    train_csv = tmp_path / "train.csv"
    train.to_csv(train_csv, index=False)

    oof0 = pd.DataFrame({SUBMISSION_ID_COL: ["s1", "s2"], **{c: [0.1, 0.9] for c in LABEL_COLS}})
    oof1 = pd.DataFrame({SUBMISSION_ID_COL: ["s3", "s4"], **{c: [0.2, 0.8] for c in LABEL_COLS}})
    p0 = tmp_path / "fold0_oof.csv"
    p1 = tmp_path / "fold1_oof.csv"
    oof0.to_csv(p0, index=False)
    oof1.to_csv(p1, index=False)

    weights = load_oof_blend_weights([p0, p1], train_csv)
    assert weights.shape == (2, len(LABEL_COLS))
    assert np.allclose(weights.sum(axis=0), 1.0)
    assert (weights >= 0).all()


def test_write_submission_schema(tmp_path):
    uids = ["a", "b"]
    probs = np.full((2, len(LABEL_COLS)), 0.5)
    path = write_submission(uids, probs, tmp_path / "sub.csv")
    df = pd.read_csv(path)
    assert list(df.columns) == [SUBMISSION_ID_COL] + LABEL_COLS


def test_load_blend_weights_json_normalizes(tmp_path):
    from rsna_knee.infer import load_blend_weights_json

    raw = np.ones((5, len(LABEL_COLS)), dtype=np.float64) * 2.0
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"weights": raw.tolist(), "labels": list(LABEL_COLS)}))
    w = load_blend_weights_json(path)
    assert w.shape == (5, len(LABEL_COLS))
    assert np.allclose(w.sum(axis=0), 1.0)


def test_baked_s02_weights_in_repo():
    from rsna_knee.infer import load_blend_weights_json

    path = Path(__file__).resolve().parents[1] / "configs" / "s02_v6c_blend_weights.json"
    assert path.exists(), "configs/s02_v6c_blend_weights.json missing"
    w = load_blend_weights_json(path)
    assert w.shape == (5, len(LABEL_COLS))
    assert np.allclose(w.sum(axis=0), 1.0, atol=1e-5)
    assert (w > 0).all()
    # Near-uniform: no fold should dominate any label after weak-AUC weighting.
    assert (w.max(axis=0) < 0.35).all()
    assert (w.min(axis=0) > 0.10).all()
