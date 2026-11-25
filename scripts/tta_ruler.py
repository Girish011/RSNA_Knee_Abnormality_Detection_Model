#!/usr/bin/env python3
"""Read test-time plans from runpod/tta_eval.py passes on the report ruler (`said`, see oof_ruler.py).

    kaggle datasets download girishbose/rsna-knee-rp-tta-eval1-out -p outputs/runpod_results/tta-eval1 --unzip
    python scripts/tta_ruler.py --dir outputs/runpod_results/tta-eval1

Each raptor_tta_<tag>.npz holds one prediction per (window phase, mirror) pass for one checkpoint on its own held-out
fold. A plan is a set of passes averaged in probability. Printed per checkpoint, and for the rank mean of the
checkpoints that share a fold (= what the leg ships), with a paired bootstrap SE against the leg's current plan
(two plain passes, phases 0 and 0.5). Paired differences are far tighter than the +-0.002 between fold-0 runs.
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
PLANS = {   # name -> [(phase, mirrored)]
    "leg now: 2 plain (0, .5)": [(0.0, 0), (0.5, 0)],
    "1 plain (0)": [(0.0, 0)],
    "2: plain 0 + mirror .5 (free)": [(0.0, 0), (0.5, 1)],
    "2 mirrored (0, .5)": [(0.0, 1), (0.5, 1)],
    "4 plain (0, .25, .5, .75)": [(0.0, 0), (0.25, 0), (0.5, 0), (0.75, 0)],
    "4: 2 plain + 2 mirror": [(0.0, 0), (0.5, 0), (0.0, 1), (0.5, 1)],
    "4: alternate (0, .5 plain; .25, .75 mirror)": [(0.0, 0), (0.5, 0), (0.25, 1), (0.75, 1)],
    "8: all": [(p, f) for f in (0, 1) for p in (0.0, 0.25, 0.5, 0.75)],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
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

    def se(A: pd.DataFrame, B: pd.DataFrame, n: int = 200) -> float:
        rng = np.random.default_rng(0)
        ids = A.index.values
        return float(np.std([said(B.loc[s]) - said(A.loc[s]) for s in (rng.choice(ids, len(ids)) for _ in range(n))]))

    runs = {}
    for f in sorted(Path(a.dir).rglob("raptor_tta_*.npz")):
        z = np.load(f, allow_pickle=True)
        key = {(round(float(p), 2), int(fl)): i for i, (p, fl) in enumerate(zip(z["phases"], z["flips"]))}
        ids = z["ids"].astype(str)
        pred = z["pred"].astype(np.float32)
        runs[f.stem.replace("raptor_tta_", "")] = (int(z["fold"]), {
            name: pd.DataFrame(np.mean([pred[key[(round(p, 2), fl)]] for p, fl in plan], axis=0), index=ids, columns=LAB)
            for name, plan in PLANS.items()})
    base_name = next(iter(PLANS))

    def table(title: str, plans: dict[str, pd.DataFrame]) -> None:
        base = plans[base_name]
        print(f"{title} | {len(base)} studies | {base_name}: said {said(base):.4f}")
        for name, P in plans.items():
            if name != base_name:
                print(f"    {name:44s} {said(P):.4f} ({said(P) - said(base):+.4f} +-{se(base, P):.4f})")

    for tag, (fold, plans) in runs.items():
        table(f"{tag} (fold {fold})", plans)
    folds = sorted({f for f, _ in runs.values()})
    pooled = {}
    for name in PLANS:
        parts = []
        for k in folds:
            members = [pl[name] for f, pl in runs.values() if f == k]
            idx = members[0].index
            parts.append(sum(m.loc[idx].apply(lambda c: rankdata(c) / len(c)) for m in members) / len(members))
        pooled[name] = pd.concat(parts)
    table(f"rank mean of the checkpoints per fold, folds {folds} pooled", pooled)


if __name__ == "__main__":
    main()
