import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("score_definition_pilot", REPO / "scripts/score_definition_pilot.py")
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)


def _receipts():
    a = {k: "same" for k in scorer.MATCH_FIELDS}
    a.update(protocol={"checkpoint_policy": "fixed_ema", "views": "global"}, target_mode="control", aux_weight=0)
    b = {**a, "target_mode": "definition"}
    return a, b


def test_protocol_match_exposes_only_intended_change():
    a, b = _receipts()
    assert scorer.validate_receipts(a, b)["intentional_changes"] == ["target_mode"]
    b["seed"] = 99
    with pytest.raises(ValueError, match="unmatched"):
        scorer.validate_receipts(a, b)


def test_auxiliary_change_must_be_declared():
    a, b = _receipts()
    b["aux_weight"] = .1
    with pytest.raises(ValueError, match="aux_weight"):
        scorer.validate_receipts(a, b)
    assert scorer.validate_receipts(a, b, allow_aux_change=True)["intentional_changes"] == ["target_mode", "aux_weight"]


def test_best_epoch_and_missing_receipt_refused():
    a, b = _receipts()
    b["protocol"] = {"checkpoint_policy": "best_gold"}
    with pytest.raises(ValueError, match="fixed_ema"):
        scorer.validate_receipts(a, b)
    a, b = _receipts()
    b.pop("init_sha256")
    with pytest.raises(ValueError, match="missing"):
        scorer.validate_receipts(a, b)


def test_exact_ids_required_and_order_aligned():
    a = np.tile(np.array([[.1], [.9]]), (1, 12))
    c, d = scorer.align_pairs(np.array(["a", "b"]), a, np.array(["b", "a"]), a[::-1], np.array(["b", "a"]))
    np.testing.assert_array_equal(c, d)
    with pytest.raises(ValueError, match="complete"):
        scorer.align_pairs(np.array(["a", "b"]), a, np.array(["b", "a"]), a, np.array(["a", "b", "c"]))


def test_paired_bootstrap_is_deterministic_and_zero_for_identical_arms():
    y = np.tile(np.array([0] * 15 + [1] * 15)[:, None], (1, 12))
    p = np.random.default_rng(7).uniform(size=y.shape)
    m = np.ones(y.shape, bool)
    a = scorer.paired_score(y, m, p, p, n_bootstrap=40)
    b = scorer.paired_score(y, m, p, p, n_bootstrap=40)
    assert a == b and a["paired_delta_ci95"] == [0., 0.]
    assert a["resampling_unit"] == "study"


def test_masked_cells_cannot_change_metrics():
    y = np.tile(np.array([0] * 15 + [1] * 15)[:, None], (1, 12))
    p = np.random.default_rng(7).uniform(size=y.shape)
    m = np.ones(y.shape, bool)
    m[:4] = False
    q = p.copy()
    q[:4] = 1 - q[:4]
    result = scorer.paired_score(y, m, p, q, n_bootstrap=40)
    assert result["macro_delta"] == 0 and result["paired_delta_ci95"] == [0., 0.]


def test_known_auc_and_patient_cluster_label():
    y = np.tile(np.array([0] * 15 + [1] * 15)[:, None], (1, 12))
    a, b = 1 - y * .8 - .1, y * .8 + .1
    result = scorer.paired_score(y, np.ones_like(y, bool), a, b, n_bootstrap=40, groups=np.repeat(np.arange(15), 2))
    assert result["macro_delta"] == pytest.approx(1)
    assert result["paired_delta_ci95"] == [1., 1.]
    assert result["resampling_unit"] == "verified_patient_cluster"
