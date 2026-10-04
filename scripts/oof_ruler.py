#!/usr/bin/env python3
"""Large offline ruler for recipe A/Bs: fold OOF predictions (~870 studies) instead of the 29 experts.

    python scripts/oof_ruler.py --base r1-c224-f0:r1c2240 --runs r1-c224k32-f0:r1c224k320,r2-c224reg-f0:r2c224reg0

Two readings per run, each with a bootstrap SE of the difference to --base (same studies):
  said   = macro AUC vs the report where it explicitly says YES / NO (silent cells left out)
  target = macro AUC vs --labels >= 0.5 (the round target the runs trained on)
Noise is ~0.001-0.002, against 0.01-0.02 on the experts. It measures agreement with reports / teachers, not experts:
use it for recipe and backbone changes on the SAME labels; it cannot judge a label change (2026-10-04: labels_r2g read
-0.0001 on `said` but -0.017 on holdout-29). c224 vs CoAt-384 read +0.0015 here and +0.001 on the LB.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="job:tag")
    ap.add_argument("--runs", required=True, help="comma list of job:tag")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--labels", default="labels_r1.parquet")
    a = ap.parse_args()
    pk = pd.read_csv(REPO / "data/external/public_labels/pilk/report_labels_v2.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    lab = pd.read_parquet(REPO / "data/processed" / a.labels)
    lab = lab.assign(StudyInstanceUID=lab["StudyInstanceUID"].astype(str)).set_index("StudyInstanceUID")

    def oof(spec: str) -> pd.DataFrame:
        job, tag = spec.split(":")
        z = np.load(REPO / "outputs/runpod_results" / job / tag / f"raptor_oof_{tag}_fold{a.fold}.npz", allow_pickle=True)
        return pd.DataFrame(z["pred"], index=z["ids"].astype(str), columns=LAB)

    def said(P: pd.DataFrame) -> float:
        out = []
        for f in LAB:
            v = pk[f + "__verdict"].reindex(P.index)
            m = v.isin(["YES", "NO"]).values
            y = (v[m] == "YES").astype(int)
            if 0 < y.sum() < len(y):
                out.append(roc_auc_score(y, P[f][m]))
        return float(np.mean(out))

    def target(P: pd.DataFrame) -> float:
        return float(np.mean([roc_auc_score((lab.loc[P.index, f] >= .5).astype(int), P[f]) for f in LAB]))

    def se(A: pd.DataFrame, B: pd.DataFrame, fn, n: int = 200) -> float:
        rng = np.random.default_rng(0)
        ids = A.index.values
        d = []
        for _ in range(n):
            s = rng.choice(ids, len(ids))
            d.append(fn(B.loc[s]) - fn(A.loc[s]))
        return float(np.std(d))

    base = oof(a.base)
    print(f"fold {a.fold}, {len(base)} studies | base {a.base}: said {said(base):.4f}  target {target(base):.4f}")
    for spec in a.runs.split(","):
        P = oof(spec).loc[base.index]
        print(f"  {spec:34s} said {said(P):.4f} ({said(P) - said(base):+.4f} +-{se(base, P, said):.4f})  "
              f"target {target(P):.4f} ({target(P) - target(base):+.4f} +-{se(base, P, target):.4f})")


if __name__ == "__main__":
    main()
