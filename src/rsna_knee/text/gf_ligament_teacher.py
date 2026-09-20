"""Greenfield gap-fill teacher: fill NaN cells on top of weak_v1.

Design:
- Start from ``weak_labels_v1.csv`` (known training teacher).
- Only fill cells that are currently NaN for the chosen focus labels.
- Fills come from ``weak_labels_v7`` (Turkish/Greek + v2 delegate for other langs),
  after the v7/v2 pattern patches for ligament phrasing.
- Never overwrite expert gold when building the ship CSV (expert_override=True).
- Train-only; never used at inference / submit.

Recipes:
- lig1: ACL + MCL + Medial Meniscus (verdict void / noise; OOF gold 0.6019)
- lig2: Medial Meniscus only (verdict void / noise; OOF gold 0.6103; admitted new studies)
- lig3: Medial Meniscus only, restricted to studies that already have >=1 weak_v1 label
  so the training set size stays 2449 (single-factor vs lig2's study-count confound)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rsna_knee.constants import LABEL_COLS
from rsna_knee.text.weak_labels_v7 import extract_label_v7

FOCUS_LABELS_LIG1 = ("ACL", "MCL", "Medial Meniscus")
FOCUS_LABELS_LIG2 = ("Medial Meniscus",)
FOCUS_LABELS_LIG3 = ("Medial Meniscus",)
# Backward-compat alias for lig1 tests / callers.
FOCUS_LABELS = FOCUS_LABELS_LIG1


def gap_fill_focus_labels(
    base: pd.DataFrame,
    reports: pd.Series,
    *,
    min_confidence: float = 0.5,
    focus_labels: tuple[str, ...] = FOCUS_LABELS,
    eligible_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return a copy of ``base`` with NaN focus labels filled from v7 extractor.

    If ``eligible_mask`` is given (index-aligned bool Series), only those rows may
    receive fills. Use this for lig3 to avoid admitting new studies into training.
    """
    out = base.copy()
    stats = {f"{lab}_filled": 0 for lab in focus_labels}
    stats["n_rows"] = len(out)
    stats["n_eligible"] = int(eligible_mask.sum()) if eligible_mask is not None else len(out)
    stats["n_skipped_ineligible"] = 0

    for idx, row in out.iterrows():
        if eligible_mask is not None and not bool(eligible_mask.loc[idx]):
            stats["n_skipped_ineligible"] += 1
            continue
        report = reports.loc[idx] if idx in reports.index else ""
        if pd.isna(report) or not str(report).strip():
            continue
        text = str(report)
        for lab in focus_labels:
            if pd.notna(row.get(lab)):
                continue
            res = extract_label_v7(text, lab)
            conf_col = f"{lab}__conf"
            if res.confidence < min_confidence:
                out.at[idx, conf_col] = float(res.confidence)
                continue
            out.at[idx, lab] = float(res.value)
            out.at[idx, conf_col] = float(res.confidence)
            stats[f"{lab}_filled"] += 1
    return out, stats


def apply_expert_override(labels: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Overwrite with expert values wherever all 12 gold labels exist."""
    out = labels.copy()
    is_gold = train[LABEL_COLS].notna().all(axis=1)
    gold = train.loc[is_gold, ["StudyInstanceUID"] + LABEL_COLS]
    g = gold.set_index("StudyInstanceUID")
    for idx, row in out.iterrows():
        uid = str(row["StudyInstanceUID"])
        if uid not in g.index:
            continue
        for lab in LABEL_COLS:
            out.at[idx, lab] = g.at[uid, lab]
            out.at[idx, f"{lab}__conf"] = 1.0
    return out


def coverage_table(df: pd.DataFrame, labels: tuple[str, ...] = FOCUS_LABELS) -> pd.DataFrame:
    rows = []
    for lab in labels:
        s = df[lab]
        rows.append(
            {
                "label": lab,
                "known": int(s.notna().sum()),
                "pos": int((s == 1).sum()),
                "neg": int((s == 0).sum()),
            }
        )
    rows.append(
        {
            "label": "ANY",
            "known": int(df[list(LABEL_COLS)].notna().any(axis=1).sum()),
            "pos": None,
            "neg": None,
        }
    )
    return pd.DataFrame(rows)


def build_gf_gapfill(
    train_csv: Path,
    weak_v1_csv: Path,
    out_csv: Path,
    *,
    min_confidence: float = 0.5,
    focus_labels: tuple[str, ...] = FOCUS_LABELS,
    only_studies_with_any_label: bool = False,
) -> dict:
    train = pd.read_csv(train_csv)
    base = pd.read_csv(weak_v1_csv)
    rep = train.set_index("StudyInstanceUID")["Report"]
    base = base.set_index("StudyInstanceUID", drop=False)
    reports = rep.reindex(base.index)

    eligible = None
    if only_studies_with_any_label:
        eligible = base[list(LABEL_COLS)].notna().any(axis=1)

    before = coverage_table(base, labels=focus_labels)
    filled, stats = gap_fill_focus_labels(
        base,
        reports,
        min_confidence=min_confidence,
        focus_labels=focus_labels,
        eligible_mask=eligible,
    )
    filled = apply_expert_override(filled.reset_index(drop=True), train)
    after = coverage_table(filled, labels=focus_labels)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keep_upload = ["StudyInstanceUID"] + LABEL_COLS + [f"{c}__conf" for c in LABEL_COLS]
    filled[keep_upload].to_csv(out_csv, index=False)

    return {
        "stats": stats,
        "focus_labels": list(focus_labels),
        "only_studies_with_any_label": only_studies_with_any_label,
        "coverage_before": before.to_dict(orient="records"),
        "coverage_after": after.to_dict(orient="records"),
        "out_csv": str(out_csv),
        "n_studies": len(filled),
    }


def build_gf_lig1(
    train_csv: Path,
    weak_v1_csv: Path,
    out_csv: Path,
    *,
    min_confidence: float = 0.5,
) -> dict:
    return build_gf_gapfill(
        train_csv,
        weak_v1_csv,
        out_csv,
        min_confidence=min_confidence,
        focus_labels=FOCUS_LABELS_LIG1,
    )


def build_gf_lig2(
    train_csv: Path,
    weak_v1_csv: Path,
    out_csv: Path,
    *,
    min_confidence: float = 0.5,
) -> dict:
    return build_gf_gapfill(
        train_csv,
        weak_v1_csv,
        out_csv,
        min_confidence=min_confidence,
        focus_labels=FOCUS_LABELS_LIG2,
    )


def build_gf_lig3(
    train_csv: Path,
    weak_v1_csv: Path,
    out_csv: Path,
    *,
    min_confidence: float = 0.5,
) -> dict:
    return build_gf_gapfill(
        train_csv,
        weak_v1_csv,
        out_csv,
        min_confidence=min_confidence,
        focus_labels=FOCUS_LABELS_LIG3,
        only_studies_with_any_label=True,
    )
