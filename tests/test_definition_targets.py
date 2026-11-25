"""Definition-target contracts: missingness, rubric limits, isolation and provenance."""
import numpy as np
import pandas as pd
import pytest

from rsna_knee.constants import LABEL_COLS
from rsna_knee.definition_targets import (
    State, build_arrays, control_cell, definition_cell, policy_manifest, validate_arrays,
)


def _frames():
    ids = ["a", "b", "expert"]
    train = pd.DataFrame({"StudyInstanceUID": ids, "Report": ["not exported"] * 3,
                          **{l: [np.nan, np.nan, 1.] for l in LABEL_COLS}})
    grades = pd.DataFrame({"StudyInstanceUID": ids, **{l: [3, 1, 4] for l in LABEL_COLS}})
    pilk = pd.DataFrame({"StudyInstanceUID": ids, **{l: [.82, .28, .82] for l in LABEL_COLS},
                        **{l + "__verdict": ["YES", "UNK", "YES"] for l in LABEL_COLS},
                        **{l + "__conf": [.95, .05, .95] for l in LABEL_COLS}})
    dread = pd.DataFrame({"StudyInstanceUID": ids[:2], **{l: [.8, .05] for l in LABEL_COLS}})
    return train, grades, pilk, dread


def test_silent_and_missing_never_become_negatives():
    for l in LABEL_COLS:
        for grade in (1, None):
            cell = definition_cell(l, grade, "UNK", .05)
            assert not cell.mask and cell.confidence == 0
            assert not control_cell(grade, "UNK", .28, .05, .05).mask


def test_explicit_absence_and_conflict_are_distinct():
    absent = definition_cell("ACL", 0, "NO", .05)
    assert absent.mask and absent.strong and absent.target == 0
    conflict = definition_cell("ACL", 0, "YES", .8)
    assert conflict.state == State.CONFLICT and not conflict.mask
    assert not definition_cell("ACL", 4, "NO", .8).mask


def test_definition_rejects_mild_but_control_keeps_raw_report_signal():
    low = definition_cell("Effusion", 2, "YES", .85)
    assert low.mask and low.target == 0 and low.state == State.BELOW_THRESHOLD
    raw = control_cell(2, "YES", .82, .95, .85)
    assert raw.mask and raw.target == pytest.approx(.835)
    assert definition_cell("Effusion", 3, "YES", .85).target == 1


def test_missing_oa_extent_and_mcl_acuity_are_never_invented():
    for label in ("Medial OA", "Lateral OA", "PF OA", "MCL", "Baker's", "Fracture"):
        cell = definition_cell(label, 4, "YES", .85)
        assert cell.mask and cell.target == 1 and not cell.strong
        assert cell.confidence <= .65
    assert definition_cell("ACL", 4, "YES", .85).strong


def test_implied_synovitis_and_degenerative_contusion_are_unresolved():
    assert not definition_cell("Synovitis", 2, "YES", .8).mask
    assert not definition_cell("Contusion", 2, "YES", .8).mask
    assert not definition_cell("ACL", 2, "UNK", .05).mask


def test_contusion_requires_explicit_fracture_denial_to_be_strong():
    weak = definition_cell("Contusion", 3, "YES", .85, "UNK")
    compatible = definition_cell("Contusion", 3, "YES", .85, "NO")
    assert weak.mask and not weak.strong
    assert compatible.mask and compatible.strong


def test_exact_frozen_order_excludes_experts_and_masks_silent_ordinal():
    a = build_arrays(*_frames(), {"b": 3, "a": 0})
    assert list(a["ids"]) == ["b", "a"]
    assert list(a["folds"]) == [3, 0]
    assert not a["mask"][0].any() and not a["ordinal_mask"][0].any()
    np.testing.assert_array_equal(a["ordinal_target"][1], np.tile([1, 1, 0], (12, 1)))
    assert not bool(a["is_expert_diagnostic"])
    validate_arrays(a, {"expert"})


def test_gold_values_never_calibrate_or_overwrite_training_targets():
    frames = _frames()
    a = build_arrays(*frames, {"a": 0, "b": 1})
    frames[0].loc[2, LABEL_COLS] = 0.
    b = build_arrays(*frames, {"a": 0, "b": 1})
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    ex = build_arrays(*frames, {"a": 0, "b": 1}, expert_diagnostic=True)
    assert list(ex["ids"]) == ["expert"] and bool(ex["is_expert_diagnostic"])
    assert np.all(ex["folds"] == -1)


@pytest.mark.parametrize("folds", [{"a": 0}, {"a": 0, "b": 5}, {"a": 0, "b": 1, "expert": 2}])
def test_bad_folds_fail_closed(folds):
    with pytest.raises(ValueError):
        build_arrays(*_frames(), folds)


def test_duplicate_ids_and_invalid_grades_fail_closed():
    frames = list(_frames())
    frames[1] = pd.concat([frames[1], frames[1].iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        build_arrays(*frames, {"a": 0, "b": 1})
    with pytest.raises(ValueError, match="integer"):
        definition_cell("ACL", 2.5, "YES")


def test_manifest_exposes_limits_and_has_no_expert_fit():
    manifest = policy_manifest()
    assert manifest["gold_fitted_parameters"] is False
    assert manifest["forbidden_image_teacher_inputs_used"] is False
    assert "AND" in manifest["official_definition_summary"]["Lateral OA"]
    assert manifest["grade_rules"]["Lateral OA"]["4"]["definition_compatible"] is False


def test_contradictory_fracture_denial_cannot_strengthen_contusion():
    frames = list(_frames())
    frames[2].loc[0, "Fracture__verdict"] = "NO"  # grader says definite fracture
    arrays = build_arrays(*frames, {"a": 0, "b": 1})
    j = LABEL_COLS.index("Contusion")
    assert arrays["weak_mask"][0, j] and not arrays["strong_mask"][0, j]


def test_uncertain_grade2_without_confirmation_has_no_ordinal_loss():
    frames = list(_frames())
    frames[1].loc[1, "ACL"] = 2
    arrays = build_arrays(*frames, {"a": 0, "b": 1})
    assert not arrays["mask"][1, 0] and not arrays["ordinal_mask"][1, 0].any()
