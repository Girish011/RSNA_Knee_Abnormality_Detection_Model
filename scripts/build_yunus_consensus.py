#!/usr/bin/env python3
"""Build yunus∩v6c additive gap-fill candidate (train-only).

High-precision cells: label only where our v6c and yunus public labels both commit
and agree (after soft→hard with lo/hi). Overlay onto v6c only where v6c is NaN —
does not overwrite trusted v6c commits (safer than v8 overwrite variant).

Usage:
  python scripts/build_yunus_consensus.py \\
    --base data/processed/weak_labels_v6c.csv \\
    --yunus data/external/rsna-knee-llm-report-labels/yunus_llm_labels.csv \\
    --soft-lo 0.2 --soft-hi 0.7 \\
    --out data/processed/weak_labels_yunus_gap.csv \\
    --gold data/raw/train.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.constants import LABEL_COLS, SUBMISSION_ID_COL
from rsna_knee.text.label_consensus import intersect_labels, overlay_consensus


def _harden(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    out = df.copy()
    for c in LABEL_COLS:
        x = pd.to_numeric(out[c], errors="coerce")
        h = pd.Series(pd.NA, index=x.index, dtype="Float64")
        h = h.mask(x <= lo, 0.0)
        h = h.mask(x >= hi, 1.0)
        out[c] = h
    return out


def _gold_audit(labels: pd.DataFrame, gold: pd.DataFrame) -> dict[str, float]:
    g = gold.loc[gold[LABEL_COLS].notna().all(axis=1)].copy()
    m = g.merge(labels, on=SUBMISSION_ID_COL, how="inner", suffixes=("_g", "_p"))
    precs = []
    for lab in LABEL_COLS:
        p = m[f"{lab}_p"]
        y = m[f"{lab}_g"].astype(float)
        committed = p.notna()
        if not committed.any():
            continue
        pc = p[committed].astype(float)
        yc = y[committed]
        tp = int(((yc == 1) & (pc == 1)).sum())
        fp = int(((yc == 0) & (pc == 1)).sum())
        if tp + fp:
            precs.append(tp / (tp + fp))
    return {"macro_precision_committed": float(np.mean(precs)) if precs else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--yunus", type=Path, required=True)
    ap.add_argument("--soft-lo", type=float, default=0.2)
    ap.add_argument("--soft-hi", type=float, default=0.7)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gold", type=Path, default=None)
    args = ap.parse_args()

    base = pd.read_csv(args.base)
    base[SUBMISSION_ID_COL] = base[SUBMISSION_ID_COL].astype(str)
    yunus = _harden(pd.read_csv(args.yunus), args.soft_lo, args.soft_hi)
    yunus[SUBMISSION_ID_COL] = yunus[SUBMISSION_ID_COL].astype(str)

    consensus = intersect_labels(base, yunus)
    out = overlay_consensus(base, consensus, only_where_base_isna=True)

    known_before = int(base[LABEL_COLS].notna().sum().sum())
    known_after = int(out[LABEL_COLS].notna().sum().sum())
    print(f"known cells: {known_before} -> {known_after} (+{known_after - known_before})")

    both = int(consensus[LABEL_COLS].notna().sum().sum())
    agree = both  # intersect only keeps agreement
    print(f"yunus∩v6c agreement cells: {agree}")

    if args.gold is not None:
        gold = pd.read_csv(args.gold)
        p_base = _gold_audit(base, gold)
        p_out = _gold_audit(out, gold)
        print(f"gold macro prec (committed): base={p_base['macro_precision_committed']:.3f} out={p_out['macro_precision_committed']:.3f}")

    keep = [SUBMISSION_ID_COL, *LABEL_COLS]
    for c in LABEL_COLS:
        conf = f"{c}__conf"
        if conf in out.columns:
            keep.append(conf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out[keep].to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
