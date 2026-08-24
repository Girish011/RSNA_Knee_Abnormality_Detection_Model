#!/usr/bin/env python3
"""Report full-OOF macro AUC (with bootstrap CI) and apply the pre-registered rule.

Examples
--------
Single run, full 5-fold OOF vs the committed folds (expert labels where present):

    python scripts/oof_report.py \
        --oof outputs/experiments/main_dinov2_b_v1/fold*_oof.csv \
        --targets data/folds/folds_v1.csv

A/B a candidate against a baseline with the keep/kill rule:

    python scripts/oof_report.py \
        --oof outputs/experiments/labels_v3_robust/fold*_oof.csv \
        --baseline-oof outputs/experiments/main_dinov2_b_v1/fold*_oof.csv \
        --targets data/folds/folds_v1.csv --margin 0.005
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd

from rsna_knee.constants import LABEL_COLS
from rsna_knee.evaluation import decide, evaluate_oof, load_oof


def _print_summary(name: str, summary: dict) -> None:
    b = summary["bootstrap"]
    print(f"\n== {name} ==")
    print(f"studies={summary['n_studies']}  macro_auc={summary['macro_auc']:.4f}")
    print(f"bootstrap mean={b['mean']:.4f}  95% CI=[{b['lo']:.4f}, {b['hi']:.4f}]  std={b['std']:.4f}")
    per = summary["per_label_auc"]
    ns = summary["n_supervised"]
    print("per-label AUC (n supervised):")
    for c in LABEL_COLS:
        v = per[c]
        vs = "  nan" if math.isnan(v) else f"{v:.3f}"
        print(f"  {c:<18} {vs}  (n={ns[c]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", nargs="+", required=True, help="candidate per-fold OOF CSVs")
    ap.add_argument("--baseline-oof", nargs="+", default=None, help="baseline per-fold OOF CSVs")
    ap.add_argument("--targets", type=Path, required=True, help="frame with StudyInstanceUID + label cols")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.005, help="pre-registered keep margin")
    args = ap.parse_args()

    targets = pd.read_csv(args.targets)
    cand = evaluate_oof(load_oof(args.oof), targets, n_boot=args.n_boot, seed=args.seed)
    _print_summary("candidate", cand)

    if args.baseline_oof:
        base = evaluate_oof(load_oof(args.baseline_oof), targets, n_boot=args.n_boot, seed=args.seed)
        _print_summary("baseline", base)
        d = decide(
            cand["macro_auc"],
            base["macro_auc"],
            margin=args.margin,
            candidate_ci=(cand["bootstrap"]["lo"], cand["bootstrap"]["hi"]),
            baseline_ci=(base["bootstrap"]["lo"], base["bootstrap"]["hi"]),
        )
        print(f"\n== DECISION: {d.verdict.upper()} ==")
        print(f"delta={d.delta:+.4f}  margin={d.margin:.4f}")
        print(d.rationale)


if __name__ == "__main__":
    main()
