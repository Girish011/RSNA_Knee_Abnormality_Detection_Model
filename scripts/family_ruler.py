#!/usr/bin/env python3
"""Five-fold OOF ruler for model families and their rank blends (`said` = macro AUC vs explicit report YES / NO).

    python scripts/family_ruler.py [--folds 0,1,2,3,4] [--blend c224,effv2s --blend effv2s,swin ...]

Families are the FAM table below (one job:tag pattern per fold). Predictions are rank-normalised inside each fold
before pooling, blends are rank means. Noise ~+-0.0005 on 4,349 studies. Same limits as oof_ruler.py: it scores
agreement with the reports, so it ranks recipes on the same labels and may misjudge families against the LB.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
FAM = {   # name -> (job pattern, tag pattern), {k} = fold; fold-0 exceptions below
    "c224": ("r1-c224-f{k}", "r1c224{k}"), "coat384": ("r1-coat-f{k}", "r1coat{k}"), "r50": ("r1-r50-f{k}", "r1r50{k}"),
    "c224hvy": ("r2-c224hvy40-f{k}", "r2c224hvy40{k}"), "effv2s": ("r2-effv2s336-f{k}", "r2effv2s336{k}"),
    "swin": ("r2-swint224-f{k}", "r2swint224{k}"), "dinos": ("r2-dinos336-f{k}", "r2dinos336{k}"),
    "sex50": ("r2-sex50336-f{k}", "r2sex50336{k}"), "cnxs": ("r2-cnxs336-f{k}", "r2cnxs336{k}"),
    "cnxs4": ("r4-cnxs336-f{k}", "r4cnxs336{k}"),   # ConvNeXt-S second-round students (labels_r4)
}
ALT = {("c224hvy", 0): "r2-c224hvy40-f0b"}


def load(fam: str, k: int) -> pd.DataFrame | None:
    job, tag = (p.format(k=k) for p in FAM[fam])
    job = ALT.get((fam, k), job)
    f = REPO / "outputs" / "runpod_results" / job / tag / f"raptor_oof_{tag}_fold{k}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    P = pd.DataFrame(z["pred"], index=z["ids"].astype(str), columns=LAB)
    return P.apply(lambda c: rankdata(c) / len(c))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--blend", action="append", default=[])
    a = ap.parse_args()
    folds = [int(x) for x in a.folds.split(",")]
    pk = pd.read_csv(REPO / "data/external/public_labels/pilk/report_labels_v2.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")

    def said(P: pd.DataFrame) -> float:
        out = []
        for f in LAB:
            v = pk[f + "__verdict"].reindex(P.index)
            m = v.isin(["YES", "NO"]).values
            y = (v[m] == "YES").astype(int)
            if 0 < y.sum() < len(y):
                out.append(roc_auc_score(y, P[f][m]))
        return float(np.mean(out))

    have = {fam: {k: load(fam, k) for k in folds} for fam in FAM}
    full = [fam for fam in FAM if all(have[fam][k] is not None for k in folds)]
    part = {fam: [k for k in folds if have[fam][k] is not None] for fam in FAM if fam not in full}
    pooled = {fam: pd.concat([have[fam][k] for k in folds]) for fam in full}
    n = len(next(iter(pooled.values()))) if pooled else 0
    print(f"folds {folds} | {n} studies | complete: {full} | partial: { {k: v for k, v in part.items() if v} }")
    for fam in full:
        print(f"  {fam:10s} said {said(pooled[fam]):.4f}")
    for spec in a.blend:
        parts = spec.split(",")
        if any(p not in pooled for p in parts):
            print(f"  blend {spec}: missing folds for {[p for p in parts if p not in pooled]}")
            continue
        idx = pooled[parts[0]].index
        B = sum(pooled[p].loc[idx] for p in parts) / len(parts)
        print(f"  blend {'+'.join(parts):40s} said {said(B):.4f}")


if __name__ == "__main__":
    main()
