"""Consensus weak labels: keep the keyword skeleton; LLM fills unlabeled cells only.

Train-only. Never used at inference. Expert gold is applied only after the gate
passes, and never during the audit that decides the gate.
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from rsna_knee.constants import LABEL_COLS
from rsna_knee.text.weak_labels_v1 import apply_weak_labels as apply_v1_keywords

# Frozen before looking at a new model's expert audit.
POS_CONFIDENCE = 0.95
NEG_CONFIDENCE = 0.98
PRECISION_GATE = 0.69
PARSE_GATE = 0.98
# Historical pre-override keyword audit: docs/audit/weak_label_vs_expert.csv
V1_MACRO_POSITIVE_PRECISION = 0.6875534188034188
V1_MACRO_RECALL = 0.3440052912592958
MIN_NEW_KNOWN_LABELS = 50
MIN_LLM_EXPERT_POSITIVES = 8

JSON_OBJECT = re.compile(r"\{.*\}", flags=re.S)


def findings_json_schema(labels: list[str]) -> dict[str, Any]:
    """JSON Schema for constrained named-findings generation.

    Property names are the exact competition label strings, including
    ``PF OA`` and ``Baker's``. Constrained decoding must emit every requested
    key; extra keys are forbidden. Thresholds and the expert gate are unchanged.
    """
    if not labels:
        raise ValueError("labels must be non-empty")
    unknown = [label for label in labels if label not in LABEL_COLS]
    if unknown:
        raise ValueError(f"unknown labels: {unknown}")
    finding = {
        "type": "object",
        "properties": {
            "value": {"enum": [0, 1, None]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["value", "confidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "findings": {
                "type": "object",
                "properties": {label: finding for label in labels},
                "required": list(labels),
                "additionalProperties": False,
            }
        },
        "required": ["findings"],
        "additionalProperties": False,
    }


def pending_labels(row: pd.Series) -> list[str]:
    """Labels the keyword skeleton left unlabeled."""
    return [label for label in LABEL_COLS if pd.isna(row.get(label))]


def known_context(row: pd.Series) -> dict[str, int]:
    """Already-accepted skeleton labels, shown to the LLM as read-only context."""
    context: dict[str, int] = {}
    for label in LABEL_COLS:
        value = row.get(label)
        if pd.notna(value):
            context[label] = int(value)
    return context


def parse_named_findings(text: str, requested: list[str]) -> dict[str, tuple[object, float]]:
    """Parse named JSON for the requested labels only.

    Accepted shapes:
      {"findings": {"ACL": {"value": 1, "confidence": 0.96}, ...}}
      {"ACL": {"value": 1, "confidence": 0.96}, ...}
    Missing requested keys abstain. Extra keys are ignored.
    """
    match = JSON_OBJECT.search(text or "")
    if match is None:
        raise ValueError("no JSON object")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("JSON root is not an object")
    raw = obj.get("findings", obj)
    if not isinstance(raw, dict):
        raise ValueError("findings is not an object")

    parsed: dict[str, tuple[object, float]] = {}
    for label in requested:
        item = raw.get(label)
        if item is None:
            parsed[label] = (None, 0.0)
            continue
        if not isinstance(item, dict):
            raise ValueError(f"{label} is not an object")
        value = item.get("value", item.get("label"))
        if value in ("null", "None", ""):
            value = None
        if isinstance(value, bool):
            raise ValueError(f"boolean not allowed for {label}: {value!r}")
        if value in (0, 1, 0.0, 1.0):
            value = int(value)
        elif value is not None:
            raise ValueError(f"invalid value for {label}: {value!r}")
        conf = float(item.get("confidence", 0.0))
        if not np.isfinite(conf) or not 0.0 <= conf <= 1.0:
            raise ValueError(f"invalid confidence for {label}: {item.get('confidence')!r}")
        parsed[label] = (value, 0.0 if value is None else conf)
    return parsed


def accept_fill(value: object, confidence: float) -> float:
    """Map an LLM cell to 1 / 0 / NaN using frozen thresholds."""
    if value == 1 and confidence >= POS_CONFIDENCE:
        return 1.0
    if value == 0 and confidence >= NEG_CONFIDENCE:
        return 0.0
    return float("nan")


def skeleton_without_expert_leak(v1: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Keep shipped v1 labels, but strip gold overrides on the 58 experts.

    Distributed weak_v1 already contains expert gold. Using it as the fill
    skeleton would hide the cells the LLM is supposed to recover and would
    leak gold into the gate audit.
    """
    out = v1.copy()
    out["StudyInstanceUID"] = out["StudyInstanceUID"].astype(str)
    train = train.copy()
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    expert_ids = train.loc[train[LABEL_COLS].notna().all(axis=1), "StudyInstanceUID"]
    if expert_ids.empty:
        return out.reset_index(drop=True)

    cols = ["StudyInstanceUID", "Report"] + [c for c in LABEL_COLS if c in train.columns]
    expert_rows = train.loc[train["StudyInstanceUID"].isin(expert_ids), cols].copy()
    keyword = apply_v1_keywords(expert_rows, min_confidence=0.5, expert_override=False)
    keyword["StudyInstanceUID"] = keyword["StudyInstanceUID"].astype(str)
    out = out.set_index("StudyInstanceUID")
    keyword = keyword.set_index("StudyInstanceUID")
    for uid in keyword.index:
        if uid not in out.index:
            out.loc[uid, :] = np.nan
        for label in LABEL_COLS:
            out.at[uid, label] = keyword.at[uid, label]
            conf_col = f"{label}__conf"
            if conf_col not in out.columns:
                out[conf_col] = np.nan
            out.at[uid, conf_col] = keyword.at[uid, conf_col]
    return out.reset_index()


def count_known(df: pd.DataFrame) -> int:
    return int(df[LABEL_COLS].notna().sum().sum())


def combine_skeleton_and_fills(skeleton: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    """Preserve every non-null skeleton cell; copy LLM fills only into NaNs."""
    out = skeleton.copy()
    out["StudyInstanceUID"] = out["StudyInstanceUID"].astype(str)
    out = out.set_index("StudyInstanceUID")
    llm = fills.copy()
    if llm.empty:
        return out.reset_index()
    llm["StudyInstanceUID"] = llm["StudyInstanceUID"].astype(str)
    llm = llm.set_index("StudyInstanceUID")
    for uid in llm.index:
        if uid not in out.index:
            out.loc[uid, :] = np.nan
        for label in LABEL_COLS:
            conf_col = f"{label}__conf"
            if label not in out.columns:
                out[label] = np.nan
            if conf_col not in out.columns:
                out[conf_col] = np.nan
            if pd.isna(out.at[uid, label]) and label in llm.columns and pd.notna(llm.at[uid, label]):
                out.at[uid, label] = llm.at[uid, label]
                if conf_col in llm.columns:
                    out.at[uid, conf_col] = llm.at[uid, conf_col]
    return out.reset_index()


def llm_fill_only(skeleton: pd.DataFrame, combined: pd.DataFrame) -> pd.DataFrame:
    """Keep combined values only in cells the skeleton left unlabeled."""
    sk = skeleton.copy()
    sk["StudyInstanceUID"] = sk["StudyInstanceUID"].astype(str)
    sk = sk.set_index("StudyInstanceUID")
    comb = combined.copy()
    comb["StudyInstanceUID"] = comb["StudyInstanceUID"].astype(str)
    comb = comb.set_index("StudyInstanceUID")
    out = comb.copy()
    for uid in out.index:
        if uid not in sk.index:
            continue
        for label in LABEL_COLS:
            if pd.notna(sk.at[uid, label]):
                out.at[uid, label] = np.nan
    return out.reset_index()


def audit_source(name: str, pred: pd.DataFrame, expert: pd.DataFrame) -> pd.DataFrame:
    left = expert.copy()
    right = pred.copy()
    left["StudyInstanceUID"] = left["StudyInstanceUID"].astype(str)
    right["StudyInstanceUID"] = right["StudyInstanceUID"].astype(str)
    merged = left.merge(right, on="StudyInstanceUID", suffixes=("_gold", "_pred"))
    rows: list[dict[str, Any]] = []
    for label in LABEL_COLS:
        gold = merged[f"{label}_gold"].astype(int)
        prediction = merged[f"{label}_pred"]
        known = prediction.notna()
        positive = known & prediction.eq(1)
        if known.any():
            f1 = float(
                f1_score(
                    gold[known].to_numpy(),
                    prediction[known].astype(int).to_numpy(),
                    zero_division=0,
                )
            )
        else:
            f1 = float("nan")
        n_pos = int(positive.sum())
        rows.append(
            {
                "source": name,
                "label": label,
                "n_known": int(known.sum()),
                "n_pred_pos": n_pos,
                "positive_precision": (
                    float(precision_score(gold[positive], np.ones(n_pos), zero_division=0))
                    if n_pos
                    else float("nan")
                ),
                "positive_recall_all": float(
                    recall_score(gold.to_numpy(), positive.astype(int).to_numpy(), zero_division=0)
                ),
                "f1_known": f1,
            }
        )
    return pd.DataFrame(rows)


def _macro(series: pd.Series) -> float:
    valid = series.dropna()
    return float(valid.mean()) if len(valid) else 0.0


def evaluate_gate(
    *,
    skeleton_audit: pd.DataFrame,
    combined_audit: pd.DataFrame,
    fill_audit: pd.DataFrame,
    parse_rate: float,
    skeleton_known: int,
    combined_known: int,
) -> dict[str, Any]:
    """Precision-first gate. Recall must rise without dropping v1 precision."""
    combined_prec = _macro(combined_audit["positive_precision"])
    combined_rec = _macro(combined_audit["positive_recall_all"])
    skeleton_prec = _macro(skeleton_audit["positive_precision"])
    skeleton_rec = _macro(skeleton_audit["positive_recall_all"])
    fill_prec = _macro(fill_audit["positive_precision"])
    fill_pos = int(fill_audit["n_pred_pos"].sum())
    min_precision = max(PRECISION_GATE, V1_MACRO_POSITIVE_PRECISION, skeleton_prec)
    passed = (
        combined_prec >= min_precision
        and combined_rec > max(V1_MACRO_RECALL, skeleton_rec)
        and parse_rate >= PARSE_GATE
        and combined_known >= skeleton_known + MIN_NEW_KNOWN_LABELS
        and fill_pos >= MIN_LLM_EXPERT_POSITIVES
        and (fill_prec >= PRECISION_GATE if fill_pos else False)
    )
    return {
        "skeleton_macro_positive_precision": skeleton_prec,
        "skeleton_macro_recall": skeleton_rec,
        "combined_macro_positive_precision": combined_prec,
        "combined_macro_recall": combined_rec,
        "llm_fill_macro_positive_precision": fill_prec,
        "llm_fill_expert_positives": fill_pos,
        "parse_rate": float(parse_rate),
        "skeleton_known_labels": int(skeleton_known),
        "combined_known_labels": int(combined_known),
        "min_precision": float(min_precision),
        "min_recall": float(max(V1_MACRO_RECALL, skeleton_rec)),
        "passed": bool(passed),
    }


def fills_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert resume/progress records into an LLM fill table."""
    if not records:
        cols = ["StudyInstanceUID", "__parse_ok", "__raw_response"] + LABEL_COLS
        cols += [f"{label}__conf" for label in LABEL_COLS]
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, Any]] = []
    for rec in records:
        row: dict[str, Any] = {
            "StudyInstanceUID": str(rec["StudyInstanceUID"]),
            "__parse_ok": bool(rec.get("parse_ok", False)),
            "__raw_response": rec.get("raw_response", ""),
        }
        findings = rec.get("findings") or {}
        pending = rec.get("pending") or list(findings)
        for label in LABEL_COLS:
            row[label] = np.nan
            row[f"{label}__conf"] = np.nan
        if rec.get("parse_ok"):
            for label in pending:
                item = findings.get(label) or {}
                value = item.get("value")
                conf = float(item.get("confidence", 0.0) or 0.0)
                if value in (0, 1, 0.0, 1.0):
                    value = int(value)
                row[label] = accept_fill(value, conf)
                row[f"{label}__conf"] = conf
        rows.append(row)
    return pd.DataFrame(rows)


def apply_expert_override(candidate: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    out = candidate.copy()
    out["StudyInstanceUID"] = out["StudyInstanceUID"].astype(str)
    out = out.set_index("StudyInstanceUID")
    expert = train.loc[train[LABEL_COLS].notna().all(axis=1)].copy()
    expert["StudyInstanceUID"] = expert["StudyInstanceUID"].astype(str)
    expert = expert.set_index("StudyInstanceUID")
    for uid, row in expert.iterrows():
        if uid not in out.index:
            out.loc[uid, :] = np.nan
        for label in LABEL_COLS:
            out.at[uid, label] = float(row[label])
            out.at[uid, f"{label}__conf"] = 1.0
    keep = LABEL_COLS + [f"{label}__conf" for label in LABEL_COLS]
    return out.reset_index()[["StudyInstanceUID"] + keep]
