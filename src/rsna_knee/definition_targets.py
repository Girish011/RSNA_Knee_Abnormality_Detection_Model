"""Fixed, report-only supervision; reliability weights are NOT expert probabilities.

No fitted calibration, public image predictions, model checkpoints, network calls, or
expert labels enter target construction. Compressed report grades do not establish
every official imaging prerequisite. Such positives stay explicitly weak; silent,
uncertain and contradictory cells never become implicit negatives.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.constants import LABEL_COLS

SCHEMA_VERSION = "definition-report-v1"
RUBRIC_URL = "https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343"
RUBRIC = {
    "ACL": ">50% fiber disruption: high-grade partial or complete tear; degeneration alone negative",
    "MCL": "acute high-grade partial/complete tear, fiber disruption and edema; low-grade/chronic negative",
    "Medial Meniscus": "definite surface-contact tear on >=2 images or abnormal morphology/displaced fragment",
    "Lateral Meniscus": "definite surface-contact tear on >=2 images or abnormal morphology/displaced fragment",
    "Medial OA": "high-grade cartilage loss (>50% thickness) AND moderate/large extent (~>=1 cm)",
    "Lateral OA": "high-grade cartilage loss (>50% thickness) AND moderate/large extent (~>=1 cm)",
    "PF OA": "high-grade cartilage loss (>50% thickness) AND moderate/large extent (~>=1 cm)",
    "Effusion": "moderate/large fluid distending the joint",
    "Synovitis": "inflammation and thickening of the synovial lining",
    "Baker's": "moderate/large characteristic posterior collection",
    "Contusion": "impact marrow edema without a discrete fracture line",
    "Fracture": "acute cortical break or fracture line",
}


class State(IntEnum):
    UNKNOWN = 0
    SILENT = 1
    ABSENT = 2
    PRESENT = 3
    UNCERTAIN = 4
    CONFLICT = 5
    BELOW_THRESHOLD = 6


@dataclass(frozen=True)
class Cell:
    target: float = 0.0
    mask: bool = False
    confidence: float = 0.0
    state: State = State.UNKNOWN
    strong: bool = False
    reason: str = "missing_sources"


# (target, reliability weight, definition-compatible evidence, reason).
# These weights are fixed conservative heuristics, never fitted to the 58 experts.
# grade=0 (explicit absence) is handled separately. Grade=1 means not mentioned.
GRADE_RULES = {
    "ACL": {2: (0., .60, False, "low_grade_or_degenerative"),
            3: (1., .40, False, "high_grade_or_probable_ambiguous"),
            4: (1., .90, True, "complete_tear_reported")},
    "MCL": {2: (0., .65, False, "grade1_or_edema_only"),
            3: (1., .25, False, "partial_grade2_acuity_extent_missing"),
            4: (1., .45, False, "complete_tear_acuity_edema_unrecorded")},
    "Medial Meniscus": {2: (0., .55, False, "degeneration_fraying_or_doubtful"),
                        3: (1., .60, False, "definite_tear_image_criteria_unrecorded"),
                        4: (1., .75, False, "complex_morphology_compressed_state")},
    "Lateral Meniscus": {2: (0., .55, False, "degeneration_fraying_or_doubtful"),
                         3: (1., .60, False, "definite_tear_image_criteria_unrecorded"),
                         4: (1., .75, False, "complex_morphology_compressed_state")},
    "Medial OA": {2: (0., .60, False, "low_grade_cartilage"),
                  3: (1., .15, False, "oa_extent_and_depth_unrecorded"),
                  4: (1., .30, False, "oa_severe_or_subchondral_extent_missing")},
    "Lateral OA": {2: (0., .60, False, "low_grade_or_generic_degeneration"),
                   3: (1., .15, False, "oa_extent_and_depth_unrecorded"),
                   4: (1., .30, False, "oa_severe_or_subchondral_extent_missing")},
    "PF OA": {2: (0., .60, False, "low_grade_cartilage"),
              3: (1., .15, False, "oa_extent_and_depth_unrecorded"),
              4: (1., .30, False, "oa_severe_or_subchondral_extent_missing")},
    "Effusion": {2: (0., .75, True, "small_or_physiologic_fluid"),
                 3: (1., .85, True, "moderate_effusion_reported"),
                 4: (1., .75, False, "large_or_lipohemarthrosis_compressed_state")},
    "Synovitis": {3: (1., .55, False, "synovitis_or_hypertrophy_compressed_state"),
                  4: (1., .70, False, "marked_nodular_or_proliferative_synovitis")},
    "Baker's": {2: (0., .70, True, "small_collection_below_threshold"),
                3: (1., .10, False, "definite_cyst_size_unrecorded"),
                4: (1., .35, False, "large_or_septated_or_ruptured_size_ambiguous")},
    "Contusion": {3: (1., .55, False, "bone_bruise_fracture_line_unrecorded"),
                  4: (1., .65, False, "extensive_contusion_fracture_line_unrecorded")},
    "Fracture": {2: (0., .45, False, "doubtful_old_or_irregularity"),
                 3: (1., .50, False, "fracture_diagnosis_acuity_line_unrecorded"),
                 4: (1., .65, False, "displaced_depressed_or_comminuted_acuity_unrecorded")},
}


def _finite(value: object) -> float | None:
    try:
        v = float(value)
    except (ValueError, TypeError):
        return None
    return v if np.isfinite(v) else None


def _grade(value: object) -> int | None:
    v = _finite(value)
    if v is None:
        return None
    if v != int(v) or not 0 <= v <= 4:
        raise ValueError(f"grade must be an integer 0..4, got {value!r}")
    return int(v)


def _verdict(value: object) -> str:
    if value is None or pd.isna(value):
        return "UNK"
    v = str(value)
    if v not in {"YES", "NO", "UNK"}:
        raise ValueError(f"invalid reader verdict {v!r}")
    return v


def _conflict(grade: int | None, verdict: str, dread: float | None) -> str | None:
    if grade == 0 and verdict == "YES":
        return "graded_absent_reader_present"
    if grade is not None and grade >= 2 and verdict == "NO":
        return "graded_finding_reader_absent"
    # Dread has no absent/silent distinction: a low probability must NEVER create
    # an absence assertion. A confident positive can challenge an explicit denial.
    if dread is not None and dread >= .75 and (grade == 0 or verdict == "NO"):
        return "explicit_absence_second_reader_positive"
    return None


def definition_cell(label: str, grade: object, verdict: object, dread: object = None,
                    fracture_verdict: object = "UNK") -> Cell:
    """Encode fixed evidence rules without reading expert outcomes or report text."""
    g, v, d = _grade(grade), _verdict(verdict), _finite(dread)
    conflict = _conflict(g, v, d)
    if conflict:
        return Cell(state=State.CONFLICT, reason=conflict)
    if g == 0 or v == "NO":
        weight = .95 if g == 0 and v == "NO" else .70
        return Cell(0., True, weight, State.ABSENT, True, "explicit_report_absence")
    if g is None or g == 1:
        state = State.UNCERTAIN if v == "YES" else State.SILENT if g == 1 else State.UNKNOWN
        return Cell(state=state, reason="positive_reader_no_definition_grade" if v == "YES" else "not_mentioned_or_missing")
    rule = GRADE_RULES[label].get(g)
    if rule is None:
        return Cell(state=State.UNCERTAIN, reason="mild_or_implied_evidence_insufficient")
    # Grade 2 explicitly mixes doubtful and low-grade states. Without a second
    # explicit present reading, its meaning is unresolved, even for a negative.
    if g == 2 and v != "YES":
        return Cell(state=State.UNCERTAIN, reason="low_grade_or_doubtful_without_confirmation")
    target, weight, strong, reason = rule
    if label == "Contusion" and _verdict(fracture_verdict) == "NO":
        weight, strong, reason = .75, True, "contusion_with_explicit_fracture_denial"
    # Agreement changes reliability, never supplies an absent prerequisite.
    if v == "UNK":
        weight *= .75
    if d is not None:
        if (target == 1. and d <= .15) or (target == 0. and d >= .75 and g != 2):
            weight *= .60
        elif .15 < d < .75:
            weight *= .85
    return Cell(target, True, weight, State.PRESENT if target else State.BELOW_THRESHOLD, strong, reason)


def control_cell(grade: object, verdict: object, pilk_prob: object,
                 pilk_conf: object, dread: object = None) -> Cell:
    """Raw report-probability control; same source archive, no image teacher.

    Dread contributes only where an explicit reader/grade already establishes a
    known report finding. Its low values alone never label silent cells negative.
    """
    g, v, p, c, d = _grade(grade), _verdict(verdict), _finite(pilk_prob), _finite(pilk_conf), _finite(dread)
    conflict = _conflict(g, v, d)
    if conflict:
        return Cell(state=State.CONFLICT, reason=conflict)
    known_absent = g == 0 or v == "NO"
    known_present = v == "YES" or (g is not None and g >= 3)
    if not (known_absent or known_present):
        state = State.UNCERTAIN if g == 2 else State.SILENT if g == 1 else State.UNKNOWN
        return Cell(state=state, reason="report_unknown_excluded")
    if v in {"YES", "NO"} and p is not None:
        value, weight = p, min(max(c if c is not None else .5, 0.), 1.)
    else:
        # Fixed public-reader probability levels, not gold-fitted calibration.
        value, weight = (.08 if known_absent else .82), .50
    if d is not None:
        value = .5 * (value + d)
        if (known_present and d <= .15) or (known_absent and d >= .75):
            weight *= .60
    return Cell(value, True, weight, State.ABSENT if known_absent else State.PRESENT,
                False, "raw_report_consensus")


def _indexed(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if "StudyInstanceUID" not in df:
        raise ValueError(f"{source}: missing StudyInstanceUID")
    out = df.copy()
    out["StudyInstanceUID"] = out.StudyInstanceUID.astype(str)
    if out.StudyInstanceUID.duplicated().any():
        raise ValueError(f"{source}: duplicate study IDs")
    return out.set_index("StudyInstanceUID")


def build_arrays(train: pd.DataFrame, grades: pd.DataFrame, pilk: pd.DataFrame,
                 dread: pd.DataFrame, folds: dict[str, int], *, expert_diagnostic: bool = False) -> dict[str, np.ndarray]:
    """Create nonexpert arrays in frozen fold-map order (or a separate expert audit).

    Gold values are inspected only to identify fully labelled expert rows; never
    to define, fit, tune, weight or overwrite report-derived targets.
    """
    tr, gr, pk, dr = (_indexed(x, n) for x, n in [(train, "train"), (grades, "grades"), (pilk, "pilk"), (dread, "dread")])
    for frame, name in [(tr, "train"), (gr, "grades"), (pk, "pilk"), (dr, "dread")]:
        missing = set(LABEL_COLS) - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing label columns {sorted(missing)}")
    gold_mask = tr[LABEL_COLS].notna().all(axis=1)
    partial = tr[LABEL_COLS].notna().any(axis=1) & ~gold_mask
    if partial.any():
        raise ValueError("partially expert-labelled rows require an explicit split policy")
    gold = set(tr.index[gold_mask])
    if set(folds) & gold:
        raise ValueError("frozen nonexpert fold map includes expert studies")
    expected = set(tr.index) - gold
    if set(folds) != expected:
        raise ValueError("frozen fold-map IDs must exactly match all nonexpert training studies")
    if any(not isinstance(v, (int, np.integer)) or v not in range(5) for v in folds.values()):
        raise ValueError("frozen folds must be integers 0..4")
    ids = list(tr.index[gold_mask]) if expert_diagnostic else list(folds)
    n, k = len(ids), len(LABEL_COLS)
    a = {"ids": np.asarray(ids, dtype=str), "labels": np.asarray(LABEL_COLS, dtype=str),
         "folds": np.asarray([-1 if expert_diagnostic else folds[u] for u in ids], dtype=np.int8),
         "schema_version": np.asarray(SCHEMA_VERSION),
         "is_expert_diagnostic": np.asarray(expert_diagnostic)}
    for prefix in ("", "control_"):
        a[prefix + "target"] = np.zeros((n, k), np.float32)
        a[prefix + "mask"] = np.zeros((n, k), bool)
        a[prefix + "confidence"] = np.zeros((n, k), np.float32)
        a[prefix + "state"] = np.zeros((n, k), np.uint8)
        a[prefix + "reason"] = np.full((n, k), "", dtype="U64")
    a["strong_mask"] = np.zeros((n, k), bool)
    a["weak_mask"] = np.zeros((n, k), bool)
    a["grade"] = np.full((n, k), -1, np.int8)
    a["ordinal_target"] = np.zeros((n, k, 3), np.float32)
    a["ordinal_mask"] = np.zeros((n, k, 3), bool)
    a["ordinal_confidence"] = np.zeros((n, k, 3), np.float32)
    g, p, d = gr.reindex(ids), pk.reindex(ids), dr.reindex(ids)
    for i, uid in enumerate(ids):
        fracture_v = p.loc[uid].get("Fracture__verdict", "UNK")
        if _conflict(_grade(g.loc[uid, "Fracture"]), _verdict(fracture_v),
                     _finite(d.loc[uid, "Fracture"])):
            fracture_v = "UNK"  # contradictory fracture denial cannot strengthen contusion
        for j, label in enumerate(LABEL_COLS):
            grade = _grade(g.loc[uid, label])
            verdict = p.loc[uid].get(label + "__verdict", "UNK")
            probability, confidence = p.loc[uid, label], p.loc[uid].get(label + "__conf", None)
            other = d.loc[uid, label]
            for name, cell in [("", definition_cell(label, grade, verdict, other, fracture_v)),
                               ("control_", control_cell(grade, verdict, probability, confidence, other))]:
                a[name + "target"][i, j] = cell.target
                a[name + "mask"][i, j] = cell.mask
                a[name + "confidence"][i, j] = cell.confidence
                a[name + "state"][i, j] = cell.state
                a[name + "reason"][i, j] = cell.reason
                if not name:
                    a["strong_mask"][i, j] = cell.mask and cell.strong
                    a["weak_mask"][i, j] = cell.mask and not cell.strong
            if grade is not None:
                a["grade"][i, j] = grade
                # The auxiliary task predicts the documented extraction state,
                # not a gold severity score. Silence is never a low grade.
                if grade != 1 and a["state"][i, j] != State.CONFLICT and not (grade == 2 and _verdict(verdict) != "YES"):
                    a["ordinal_target"][i, j] = [grade >= t for t in (2, 3, 4)]
                    a["ordinal_mask"][i, j] = True
                    a["ordinal_confidence"][i, j] = .5 if grade == 2 else .75
    validate_arrays(a, gold if not expert_diagnostic else None)
    return a


def validate_arrays(a: dict[str, np.ndarray], forbidden_ids: set[str] | None = None) -> None:
    n, k = len(a["ids"]), len(LABEL_COLS)
    if len(set(a["ids"])) != n or (forbidden_ids and set(a["ids"]) & forbidden_ids):
        raise ValueError("duplicate or forbidden expert training IDs")
    if list(a["labels"]) != LABEL_COLS:
        raise ValueError("unexpected label order")
    for prefix in ("", "control_", "ordinal_"):
        shape = (n, k, 3) if prefix == "ordinal_" else (n, k)
        y, m, w = (a[prefix + x] for x in ("target", "mask", "confidence"))
        if y.shape != shape or m.shape != shape or w.shape != shape:
            raise ValueError(f"{prefix}target dimensions mismatch")
        if not np.isfinite(y).all() or not np.isfinite(w).all() or np.any((y < 0) | (y > 1)) or np.any((w < 0) | (w > 1)):
            raise ValueError("targets and reliability weights must be finite in [0,1]")
        if np.any(w[~m] != 0) or np.any(y[~m] != 0):
            raise ValueError("masked cells must be zero filled with zero weight")
    if not np.array_equal(a["mask"], a["strong_mask"] | a["weak_mask"]) or np.any(a["strong_mask"] & a["weak_mask"]):
        raise ValueError("strong/weak masks must partition supervised definition cells")


def coverage(a: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for j, label in enumerate(LABEL_COLS):
        m, pos = a["mask"][:, j], a["target"][:, j] >= .5
        rows.append({"label": label, "n": len(m), "positive": int((m & pos).sum()),
                     "negative": int((m & ~pos).sum()), "masked": int((~m).sum()),
                     "strong_positive": int((a["strong_mask"][:, j] & pos).sum()),
                     "weak_positive": int((a["weak_mask"][:, j] & pos).sum()),
                     "effective_positive_weight": float(a["confidence"][:, j][m & pos].sum()),
                     "effective_negative_weight": float(a["confidence"][:, j][m & ~pos].sum()),
                     "ordinal_known": int(a["ordinal_mask"][:, j, 0].sum()),
                     "control_known": int(a["control_mask"][:, j].sum()),
                     **{"state_" + s.name.lower(): int((a["state"][:, j] == s).sum()) for s in State}})
    return rows


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def policy_manifest() -> dict:
    return {"schema_version": SCHEMA_VERSION, "official_definition_url": RUBRIC_URL,
            "official_definition_summary": RUBRIC,
            "grade_rules": {l: {str(g): {"target": v[0], "reliability_weight": v[1],
                                             "definition_compatible": v[2], "reason": v[3]}
                                   for g, v in rules.items()} for l, rules in GRADE_RULES.items()},
            "state_codes": {s.name.lower(): int(s) for s in State},
            "ordinal_thresholds": [2, 3, 4],
            "ordinal_meaning": "report extraction grade >= threshold; grade 1 is missing, never severity 1",
            "uncertain_handling": "silent, unresolved and conflicting cells excluded; missing rubric prerequisites retained only as flagged weak evidence",
            "weights_meaning": "fixed reliability/loss weights; not probabilities of expert positivity; never fitted on expert labels",
            "limitations": ["compressed grades lack evidence spans, image count, cartilage extent/depth, acuity and fracture-line detail",
                            "grade 4 OA, MCL, Baker's and fracture are not certain official-definition positives",
                            "report silence cannot establish absent disease; unmentioned findings have zero loss",
                            "public report-reader disagreement is not an independent radiologist adjudication",
                            "existing expert58 are reused development diagnostics, never fresh validation"],
            "forbidden_image_teacher_inputs_used": False,
            "gold_fitted_parameters": False, "network_or_paid_api_calls": False}


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
