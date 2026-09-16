#!/usr/bin/env python3
"""Build greenfield gap-fill teachers (lig1 / lig2) and audit vs gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from rsna_knee.constants import LABEL_COLS
from rsna_knee.text.gf_ligament_teacher import (
    FOCUS_LABELS_LIG1,
    FOCUS_LABELS_LIG2,
    build_gf_gapfill,
)
from rsna_knee.text.weak_labels_v7 import extract_label_v7

ROOT = Path(__file__).resolve().parents[1]

RECIPES = {
    "lig1": {
        "focus": FOCUS_LABELS_LIG1,
        "out_name": "weak_labels_gf_lig1.csv",
        "audit_prefix": "gf_lig1",
    },
    "lig2": {
        "focus": FOCUS_LABELS_LIG2,
        "out_name": "weak_labels_gf_lig2.csv",
        "audit_prefix": "gf_lig2",
    },
}


def _audit_pure(train: pd.DataFrame, focus: tuple[str, ...]) -> pd.DataFrame:
    gold = train.loc[train[LABEL_COLS].notna().all(axis=1)].copy()
    rows = []
    for lab in focus:
        y = gold[lab].astype(int).to_numpy()
        preds, confs = [], []
        for report in gold["Report"].astype(str):
            r = extract_label_v7(report, lab)
            preds.append(r.value)
            confs.append(r.confidence)
        pred = np.asarray(preds, dtype=float)
        conf = np.asarray(confs, dtype=float)
        m = conf >= 0.5
        rows.append(
            {
                "label": lab,
                "expert_pos": int(y.sum()),
                "committed": int(m.sum()),
                "prec_ge_0.5": float(precision_score(y[m], pred[m], zero_division=0)) if m.any() else np.nan,
                "rec_ge_0.5": float(recall_score(y[m], pred[m], zero_division=0)) if m.any() else np.nan,
                "f1_ge_0.5": float(f1_score(y[m], pred[m], zero_division=0)) if m.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", choices=sorted(RECIPES), default="lig2")
    args = parser.parse_args()
    spec = RECIPES[args.recipe]
    focus = spec["focus"]

    train_csv = Path("/Users/girish11/Downloads/train.csv")
    if not train_csv.exists():
        train_csv = ROOT / "data" / "raw" / "train.csv"
    weak_v1 = ROOT / "data" / "processed" / "weak_labels_v1.csv"
    out_csv = ROOT / "data" / "processed" / spec["out_name"]
    audit_dir = ROOT / "docs" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_csv)
    pure = _audit_pure(train, focus)
    pure.to_csv(audit_dir / f"{spec['audit_prefix']}_pure_extractor_vs_expert.csv", index=False)
    print(f"=== {args.recipe} pure extractor (v7+patches) on gold-58, conf>=0.5 ===")
    print("focus:", ", ".join(focus))
    print(pure.to_string(index=False))

    summary = build_gf_gapfill(train_csv, weak_v1, out_csv, focus_labels=focus)
    print("\n=== gap-fill stats ===")
    print(json.dumps(summary["stats"], indent=2))
    print("\ncoverage before:")
    print(pd.DataFrame(summary["coverage_before"]).to_string(index=False))
    print("\ncoverage after:")
    print(pd.DataFrame(summary["coverage_after"]).to_string(index=False))

    ship = pd.read_csv(out_csv)
    gold = train.loc[train[LABEL_COLS].notna().all(axis=1), ["StudyInstanceUID"] + LABEL_COLS]
    m = gold.merge(ship, on="StudyInstanceUID", suffixes=("_g", "_s"))
    ok = True
    for lab in focus:
        if not np.allclose(m[f"{lab}_g"].astype(float), m[f"{lab}_s"].astype(float)):
            ok = False
            print("OVERRIDE FAIL", lab)
    print("expert override ok on focus labels:", ok)

    # lig2 must not change ACL/MCL vs weak_v1 except expert gold override.
    if args.recipe == "lig2":
        v1 = pd.read_csv(weak_v1)
        gold_uids = set(gold["StudyInstanceUID"].astype(str))
        cmp = v1.merge(ship, on="StudyInstanceUID", suffixes=("_v1", "_s"))
        for lab in ("ACL", "MCL"):
            non_gold = ~cmp["StudyInstanceUID"].astype(str).isin(gold_uids)
            left = cmp.loc[non_gold, f"{lab}_v1"]
            right = cmp.loc[non_gold, f"{lab}_s"]
            same = left.isna().eq(right.isna()) & (
                left.fillna(-1).astype(float).eq(right.fillna(-1).astype(float))
            )
            n_diff = int((~same).sum())
            print(f"non-gold {lab} cells identical to weak_v1: {n_diff == 0} (diffs={n_diff})")
            if n_diff:
                ok = False

    print("wrote", out_csv)
    (audit_dir / f"{spec['audit_prefix']}_build_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
