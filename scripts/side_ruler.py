#!/usr/bin/env python3
"""Per-finding `said` AUC and side / compartment confusion of fold OOF predictions (same files as oof_ruler.py).

    python scripts/side_ruler.py --runs r4-cnxs336-f0:r4cnxs3360,r4-cnxsseq-f0:r4cnxsseq0 [--fold 0] [--blend]

`said` = AUC vs explicit report YES / NO. Confusion rows: AUC of a finding inside the studies where its opposite-side
(or neighbouring) finding is report-YES - the subset a bag-of-windows model gets wrong (6 Oct: Lateral Meniscus 0.856
where Medial Meniscus is YES vs 0.912 where NO). The first run is the base; differences are run minus base.
--blend adds the rank mean of all runs as a last column.
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
PAIRS = [("Lateral Meniscus", "Medial Meniscus"), ("Medial Meniscus", "Lateral Meniscus"), ("Lateral OA", "Medial OA"),
         ("Medial OA", "Lateral OA"), ("PF OA", "Medial OA"), ("MCL", "ACL"), ("Synovitis", "Effusion"),
         ("Fracture", "Contusion")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma list of job:tag (first = base)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--blend", action="store_true")
    a = ap.parse_args()
    pk = pd.read_csv(REPO / "data/external/public_labels/pilk/report_labels_v2.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    P = {}
    for spec in a.runs.split(","):
        job, tag = spec.split(":")
        z = np.load(REPO / "outputs/runpod_results" / job / tag / f"raptor_oof_{tag}_fold{a.fold}.npz", allow_pickle=True)
        P[tag] = pd.DataFrame(z["pred"], index=z["ids"].astype(str), columns=LAB).apply(lambda c: rankdata(c) / len(c))
    idx = next(iter(P.values())).index
    if a.blend and len(P) > 1:
        P["blend"] = sum(p.loc[idx] for p in P.values()) / len(P)
    V = pd.DataFrame({f: pk[f + "__verdict"].reindex(idx) for f in LAB})

    def auc(p, f, mask=None):
        m = V[f].isin(["YES", "NO"]) & (True if mask is None else mask)
        y = (V[f][m] == "YES").astype(int)
        return roc_auc_score(y, p.loc[idx, f][m]) if 0 < y.sum() < len(y) else np.nan

    rows = {f: [auc(p, f) for p in P.values()] for f in LAB}
    rows["MACRO said"] = [float(np.nanmean([rows[f][i] for f in LAB])) for i in range(len(P))]
    for x, other in PAIRS:
        rows[f"{x} | {other} YES"] = [auc(p, x, V[other] == "YES") for p in P.values()]
    rows["MEAN of the confusion rows"] = [float(np.nanmean([rows[f"{x} | {o} YES"][i] for x, o in PAIRS])) for i in range(len(P))]
    rows["Spearman MedOA~LatOA (model)"] = [p["Medial OA"].corr(p["Lateral OA"], method="spearman") for p in P.values()]
    rows["Spearman MedMen~LatMen (model)"] = [p["Medial Meniscus"].corr(p["Lateral Meniscus"], method="spearman") for p in P.values()]
    T = pd.DataFrame(rows, index=list(P)).T
    base = T.columns[0]
    out = T.copy()
    for c in T.columns[1:]:
        out[c] = [f"{v:.4f} ({v - b:+.4f})" for v, b in zip(T[c], T[base])]
    out[base] = [f"{v:.4f}" for v in T[base]]
    print(f"fold {a.fold} | {len(idx)} studies | base = {base}")
    print(out.to_string())


if __name__ == "__main__":
    main()
