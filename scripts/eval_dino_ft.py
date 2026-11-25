#!/usr/bin/env python3
"""Gate for the DINO fine-tune (fork-dino-ft-fused): paired gold OOF, parent vs fine-tuned.

    python scripts/eval_dino_ft.py outputs/runpod_results/dino-ft-fused/ft_oof.npz [--public oof.npz]

Per study, the members holding it out (4 seeds of one fold) are averaged in probability, then
macro AUC over the 58 expert-read studies (pooled across folds). Also reports agreement with the
fused labels on all non-gold holdout studies. Gate (PLAN_TO_TOP): ft > 0.8425 and ft > parent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
GATE = 0.8425


def per_finding(y, p):
    return np.array([roc_auc_score(y[:, j], p[:, j]) if len(set(y[:, j])) > 1 else np.nan
                     for j in range(y.shape[1])])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("oof")
    ap.add_argument("--public", default=None, help="public oof.npz (pilkwang/rsna-knee-weights)")
    a = ap.parse_args()
    z = np.load(a.oof, allow_pickle=True)
    ids, targets = [str(s) for s in z["ids"]], [str(t) for t in z["targets"]]
    with np.errstate(all="ignore"):
        pre, post = np.nanmean(z["pre"], 0), np.nanmean(z["post"], 0)
    done = ~np.isnan(post).any(1)
    tr = pd.read_csv(REPO / "data/raw/train.csv")
    gold = tr[tr[targets].notna().all(1)].set_index("StudyInstanceUID")[targets]
    pos = {s: i for i, s in enumerate(ids)}
    gi = [pos[s] for s in gold.index if s in pos and done[pos[s]]]
    gy = gold.loc[[ids[i] for i in gi]].values.astype(int)
    rows = {"parent": per_finding(gy, pre[gi]), "ft": per_finding(gy, post[gi])}
    if a.public:
        pz = np.load(a.public, allow_pickle=True)
        ppos = {str(s): i for i, s in enumerate(pz["ids"])}
        rows["public_oof"] = per_finding(gy, pz["pred"][[ppos[ids[i]] for i in gi]])
    fused = pd.read_csv(REPO / "outputs/label_ds/labels_fused_full.csv").set_index("StudyInstanceUID")
    fi = [i for i in np.where(done)[0] if ids[i] in fused.index]
    fy = (fused.loc[[ids[i] for i in fi], targets].values > 0.5).astype(int)
    df = pd.DataFrame(rows, index=targets).T
    df["MACRO"] = df.mean(1)
    print(f"members done: {int((~np.isnan(z['post']).all((1, 2))).sum())}/{len(z['member_ids'])}; "
          f"gold n={len(gi)}; fused n={len(fi)}")
    print(df.round(3).to_string())
    print(f"fused-agreement macro: parent {np.nanmean(per_finding(fy, pre[fi])):.4f} -> "
          f"ft {np.nanmean(per_finding(fy, post[fi])):.4f}")
    ft, par = df.loc["ft", "MACRO"], df.loc["parent", "MACRO"]
    ok = ft > GATE and ft > par
    print(json.dumps({"gate": GATE, "parent": round(par, 4), "ft": round(ft, 4), "pass": bool(ok)}))


if __name__ == "__main__":
    main()
