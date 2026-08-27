#!/usr/bin/env python3
"""In-domain cross-check of our weak labels vs competitor public label CSVs.

Zero domain shift: all sources label the same RSNA Knee train studies. Use this
to find cells where we disagree with independent public label sets (and,
optionally, with 58-expert gold) — a ruler that does not require an LB submit.

Examples
--------
  python scripts/crosscheck_labels.py \\
      --ours data/processed/weak_labels_v6c.csv \\
      --other barun=data/external/barun_labels.csv \\
      --other dread=data/external/dread_labels.csv \\
      --gold data/raw/train.csv \\
      --out docs/audit/label_crosscheck_v6c
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.constants import LABEL_COLS, SUBMISSION_ID_COL
from rsna_knee.text.label_consensus import agreement_stats


def _load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if SUBMISSION_ID_COL not in df.columns:
        raise SystemExit(f"{path}: missing {SUBMISSION_ID_COL}")
    missing = [c for c in LABEL_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path}: missing label cols {missing}")
    out = df[[SUBMISSION_ID_COL, *LABEL_COLS]].copy()
    out[SUBMISSION_ID_COL] = out[SUBMISSION_ID_COL].astype(str)
    for c in LABEL_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.drop_duplicates(SUBMISSION_ID_COL, keep="first")


def _coverage(df: pd.DataFrame) -> dict[str, float]:
    known = int(df[LABEL_COLS].notna().sum().sum())
    pos = int((df[LABEL_COLS] == 1).sum().sum())
    studies_any = int(df[LABEL_COLS].notna().any(axis=1).sum())
    return {
        "n_studies": len(df),
        "studies_with_any": studies_any,
        "known_cells": known,
        "positives": pos,
        "known_per_study": known / max(len(df), 1),
    }


def _vs_gold(labels: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Positive precision/recall on fully-labeled expert studies."""
    g = gold.loc[gold[LABEL_COLS].notna().all(axis=1), [SUBMISSION_ID_COL, *LABEL_COLS]].copy()
    g[SUBMISSION_ID_COL] = g[SUBMISSION_ID_COL].astype(str)
    m = g.merge(labels, on=SUBMISSION_ID_COL, how="inner", suffixes=("_g", "_p"))
    rows = []
    for lab in LABEL_COLS:
        y = m[f"{lab}_g"].astype(float)
        p = m[f"{lab}_p"]
        committed = p.notna()
        y_c, p_c = y[committed], p[committed].astype(float)
        tp = int(((y_c == 1) & (p_c == 1)).sum())
        fp = int(((y_c == 0) & (p_c == 1)).sum())
        fn = int(((y_c == 1) & (p_c == 0)).sum())
        tn = int(((y_c == 0) & (p_c == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        rows.append(
            {
                "label": lab,
                "n_gold": len(m),
                "n_committed": int(committed.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": prec,
                "recall": rec,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ours", type=Path, required=True, help="Our weak-label CSV")
    ap.add_argument(
        "--other",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Competitor label CSV as name=path (repeatable)",
    )
    ap.add_argument("--gold", type=Path, default=None, help="Optional train.csv with expert gold")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for audit CSVs")
    args = ap.parse_args()

    ours = _load_labels(args.ours)
    args.out.mkdir(parents=True, exist_ok=True)

    cov_rows = [{"source": "ours", **_coverage(ours)}]
    print("=== coverage ===")
    print(f"ours: {_coverage(ours)}")

    for spec in args.other:
        if "=" not in spec:
            raise SystemExit(f"--other expects NAME=PATH, got {spec!r}")
        name, path_s = spec.split("=", 1)
        other = _load_labels(Path(path_s))
        cov = _coverage(other)
        cov_rows.append({"source": name, **cov})
        print(f"{name}: {cov}")

        stats = agreement_stats(ours, other)
        stats_path = args.out / f"agree_ours_vs_{name}.csv"
        stats.to_csv(stats_path, index=False)
        both = stats["n_both_commit"].sum()
        agree = stats["n_agree"].sum()
        rate = agree / both if both else float("nan")
        print(
            f"  agreement where both commit: {agree}/{both} "
            f"({rate:.3f}); wrote {stats_path}"
        )
        # Dump disagreement cells for manual inspection (cap size).
        merged = ours.merge(other, on=SUBMISSION_ID_COL, how="inner", suffixes=("_ours", f"_{name}"))
        disag_rows = []
        for lab in LABEL_COLS:
            a = merged[f"{lab}_ours"]
            b = merged[f"{lab}_{name}"]
            mask = a.notna() & b.notna() & (a.astype(float) != b.astype(float))
            sub = merged.loc[mask, [SUBMISSION_ID_COL]].copy()
            sub["label"] = lab
            sub["ours"] = a[mask].astype(float).values
            sub["other"] = b[mask].astype(float).values
            disag_rows.append(sub)
        disag = pd.concat(disag_rows, ignore_index=True) if disag_rows else pd.DataFrame()
        disag_path = args.out / f"disagree_ours_vs_{name}.csv"
        disag.to_csv(disag_path, index=False)
        print(f"  disagreement cells: {len(disag)} → {disag_path}")

    pd.DataFrame(cov_rows).to_csv(args.out / "coverage.csv", index=False)

    if args.gold is not None:
        gold = pd.read_csv(args.gold)
        gold_audit = _vs_gold(ours, gold)
        gold_path = args.out / "ours_vs_gold.csv"
        gold_audit.to_csv(gold_path, index=False)
        macro_p = float(np.nanmean(gold_audit["precision"]))
        macro_r = float(np.nanmean(gold_audit["recall"]))
        print(f"=== ours vs gold === macro prec={macro_p:.3f} rec={macro_r:.3f} → {gold_path}")

    print(f"done → {args.out}")


if __name__ == "__main__":
    main()
