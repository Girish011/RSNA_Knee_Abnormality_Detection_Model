"""Consensus labeler: fill unlabeled cells only; never overwrite the skeleton."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rsna_knee.constants import LABEL_COLS
from rsna_knee.text.consensus_labels import (
    PRECISION_GATE,
    accept_fill,
    audit_source,
    combine_skeleton_and_fills,
    evaluate_gate,
    findings_json_schema,
    known_context,
    llm_fill_only,
    parse_named_findings,
    pending_labels,
    skeleton_without_expert_leak,
)


def _empty_row(uid: str, **labels: float) -> dict:
    row = {"StudyInstanceUID": uid, "Report": "ACL tear. No fracture."}
    for label in LABEL_COLS:
        row[label] = labels.get(label, np.nan)
        row[f"{label}__conf"] = 0.55 if label in labels else np.nan
    return row


def test_pending_and_context():
    row = pd.Series(_empty_row("s1", ACL=1.0, Effusion=0.0))
    assert pending_labels(row) == [c for c in LABEL_COLS if c not in {"ACL", "Effusion"}]
    assert known_context(row) == {"ACL": 1, "Effusion": 0}


def test_findings_json_schema_uses_exact_label_strings():
    schema = findings_json_schema(["PF OA", "Baker's", "ACL"])
    props = schema["properties"]["findings"]["properties"]
    assert list(props) == ["PF OA", "Baker's", "ACL"]
    assert schema["properties"]["findings"]["required"] == ["PF OA", "Baker's", "ACL"]
    assert schema["properties"]["findings"]["additionalProperties"] is False
    assert props["Baker's"]["properties"]["value"]["enum"] == [0, 1, None]
    dumped = json.dumps(schema)
    assert "Baker's" in dumped
    assert "PF OA" in dumped


def test_parse_named_findings_and_thresholds():
    text = """
    here you go
    {"findings": {
      "ACL": {"value": 1, "confidence": 0.96},
      "Fracture": {"value": 0, "confidence": 0.99},
      "PF OA": {"value": null, "confidence": 0.4}
    }}
    """
    parsed = parse_named_findings(text, ["ACL", "Fracture", "PF OA", "MCL"])
    assert parsed["ACL"] == (1, 0.96)
    assert parsed["Fracture"] == (0, 0.99)
    assert parsed["PF OA"] == (None, 0.0)
    assert parsed["MCL"] == (None, 0.0)
    floats = parse_named_findings(
        '{"ACL": {"value": 1.0, "confidence": 0.95}}',
        ["ACL"],
    )
    assert floats["ACL"] == (1, 0.95)
    assert accept_fill(1, 0.96) == 1.0
    assert np.isnan(accept_fill(1, 0.90))
    assert accept_fill(0, 0.99) == 0.0
    assert np.isnan(accept_fill(0, 0.90))


def test_combine_never_overwrites_skeleton():
    skeleton = pd.DataFrame([_empty_row("s1", ACL=1.0)])
    fills = pd.DataFrame([_empty_row("s1", ACL=0.0, Fracture=1.0)])
    combined = combine_skeleton_and_fills(skeleton, fills)
    row = combined.set_index("StudyInstanceUID").loc["s1"]
    assert row["ACL"] == 1.0
    assert row["Fracture"] == 1.0
    assert pd.isna(row["MCL"])


def test_llm_fill_only_masks_skeleton_cells():
    skeleton = pd.DataFrame([_empty_row("s1", ACL=1.0)])
    combined = pd.DataFrame([_empty_row("s1", ACL=1.0, Fracture=1.0)])
    fills = llm_fill_only(skeleton, combined).set_index("StudyInstanceUID").loc["s1"]
    assert pd.isna(fills["ACL"])
    assert fills["Fracture"] == 1.0


def test_evaluate_gate_requires_coverage_and_fill_positives():
    expert = pd.DataFrame(
        [{"StudyInstanceUID": "s1", **{label: 1 if label == "ACL" else 0 for label in LABEL_COLS}}]
    )
    skeleton = pd.DataFrame([_empty_row("s1")])
    combined = pd.DataFrame([_empty_row("s1", ACL=1.0)])
    sk_audit = audit_source("skeleton", skeleton, expert)
    comb_audit = audit_source("combined", combined, expert)
    fill_audit = audit_source("fill", combined, expert)
    failed = evaluate_gate(
        skeleton_audit=sk_audit,
        combined_audit=comb_audit,
        fill_audit=fill_audit,
        parse_rate=0.99,
        skeleton_known=0,
        combined_known=1,
    )
    assert failed["min_precision"] >= PRECISION_GATE
    assert failed["passed"] is False
    assert failed["combined_known_labels"] == 1


def test_skeleton_strips_expert_gold():
    train = pd.DataFrame(
        [
            {
                "StudyInstanceUID": "s1",
                "Report": "There is an ACL tear and a joint effusion.",
                **{label: 1.0 for label in LABEL_COLS},
            }
        ]
    )
    v1 = pd.DataFrame([_empty_row("s1", **{label: 1.0 for label in LABEL_COLS})])
    skeleton = skeleton_without_expert_leak(v1, train).set_index("StudyInstanceUID").loc["s1"]
    assert skeleton["ACL"] == 1.0
    assert skeleton["Effusion"] == 1.0
    assert pd.isna(skeleton["Baker's"])


def test_skeleton_uses_v1_not_v2_keywords():
    train = pd.DataFrame(
        [
            {
                "StudyInstanceUID": "s1",
                "Report": "Rupture complète du LCA avec épanchement articulaire.",
                **{label: 1.0 for label in LABEL_COLS},
            }
        ]
    )
    v1 = pd.DataFrame([_empty_row("s1", **{label: 1.0 for label in LABEL_COLS})])
    skeleton = skeleton_without_expert_leak(v1, train).set_index("StudyInstanceUID").loc["s1"]
    assert pd.isna(skeleton["ACL"])
    assert pd.isna(skeleton["Effusion"])
