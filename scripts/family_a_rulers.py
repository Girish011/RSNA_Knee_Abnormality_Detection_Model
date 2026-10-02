#!/usr/bin/env python3
"""Rulers for an own-model family vs our CoAt 5-fold (PLAN_TO_TOP 2026-10-02 night).

    python scripts/family_a_rulers.py --fa 'outputs/runpod_results/fa-smoke/*' [--tag-filter fa]

(a) holdout-29 pooled AUC: each study's holdout predictions averaged over the fold models found,
    AUC per finding vs expert labels (holdout-29 was never used by the label fusion).
(b) OOF AUC vs report-derived labels (pilk, dread; binarised at 0.5): independent of our fusion.
    Compared to CoAt on exactly the same OOF studies.
(c) Spearman rank-corr vs CoAt OOF and vs the public DINO OOF (stack component): lower = more blend
    value. Also the gain of a 50/50 rank blend with CoAt on (b).
Ship rule: (a) and (b) both beat CoAt on the weak five.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
WEAK5 = ["Lateral Meniscus", "Lateral OA", "PF OA", "Synovitis", "Contusion"]
PUB = REPO / "data" / "external" / "public_labels"


def _avg(files: list[str]) -> pd.DataFrame:
    """Mean prediction per study over npz files (pred, ids)."""
    frames = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        frames.append(pd.DataFrame(z["pred"], index=z["ids"].astype(str), columns=LAB))
    if not frames:
        return pd.DataFrame(columns=LAB)
    return pd.concat(frames).groupby(level=0).mean()


def load_family(pattern: str, oof_glob: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    hold = sorted(f for d in glob.glob(pattern) for f in glob.glob(f"{d}/raptor_holdout_*.npz"))
    oof = sorted(f for d in glob.glob(pattern) for f in glob.glob(f"{d}/{oof_glob}")
                 if not re.search(r"_(swa|ema|last|best|epens|ep\d+)_fold", f))  # primary OOF only
    return _avg(hold), _avg(oof)


def public_dino_oof() -> pd.DataFrame:
    f = REPO / "outputs" / "runpod_results" / "dino-ft-fused" / "ft_oof.npz"
    if not f.exists():
        return pd.DataFrame(columns=LAB)
    z = np.load(f, allow_pickle=True)
    pre, fold, grp = z["pre"], z["fold"], z["grp"]
    P = np.stack([pre[fold == g, i].mean(0) for i, g in enumerate(grp)])
    return pd.DataFrame(P, index=z["ids"].astype(str), columns=LAB)


def aucs(P: pd.DataFrame, Y: pd.DataFrame) -> dict[str, float]:
    ids = P.index.intersection(Y.index)
    out = {}
    for f in LAB:
        y = Y.loc[ids, f].dropna()
        y = (y >= 0.5).astype(int)
        if 0 < y.sum() < len(y):
            out[f] = float(roc_auc_score(y, P.loc[y.index, f]))
    return out


def summ(t: dict[str, float]) -> str:
    w = [t[f] for f in WEAK5 if f in t]
    per = " ".join(f"{f.split()[0][:4]}{f.split()[-1][:3]} {t[f]:.3f}" for f in WEAK5 if f in t)
    return f"macro {np.mean(list(t.values())):.4f} | weak5 {np.mean(w):.4f} | {per}"


def rank_blend(a: pd.DataFrame, b: pd.DataFrame, w: float) -> pd.DataFrame:
    ids = a.index.intersection(b.index)
    ra = a.loc[ids].rank(pct=True)
    rb = b.loc[ids].rank(pct=True)
    return (1 - w) * rb + w * ra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa", required=True, help="glob of Family A result dirs")
    ap.add_argument("--fa-oof", default="raptor_oof_*_fold[0-9].npz")
    ap.add_argument("--base", default=str(REPO / "outputs/runpod_results/f-7cdb6e8357*/*"))
    ap.add_argument("--base-oof", default="raptor_oof_*_fold[0-9].npz")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    tr = pd.read_csv(REPO / "data" / "raw" / "train.csv")
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold = tr.set_index("StudyInstanceUID")[LAB].dropna()
    hold_ids = json.loads((REPO / "agent" / "protocol" / "expert_holdout.json").read_text())
    gold_h = gold.loc[[u for u in hold_ids if u in gold.index]]

    fa_h, fa_o = load_family(a.fa, a.fa_oof)
    co_h, co_o = load_family(a.base, a.base_oof)
    dino = public_dino_oof()
    res = {}
    print(f"Family A: holdout studies {len(fa_h)}, OOF studies {len(fa_o)} | CoAt OOF {len(co_o)}")

    # (a) holdout-29
    ta, tc = aucs(fa_h, gold_h), aucs(co_h, gold_h)
    print("\n(a) holdout-29 pooled AUC vs experts")
    print(f"  FamilyA {summ(ta)}\n  CoAt    {summ(tc)}")
    if len(fa_h) and len(co_h):
        print(f"  blend50 {summ(aucs(rank_blend(fa_h, co_h, 0.5), gold_h))}")
    res["a"] = {"fa": ta, "coat": tc}

    # (b) OOF vs report labels, on the same studies
    ids = fa_o.index.intersection(co_o.index)
    res["b"] = {}
    for name, path, sep in (("pilk", PUB / "pilk" / "report_labels_v2.csv", ","),
                            ("dread", PUB / "dread" / "labels_llm_soft.csv", ",")):
        rep = pd.read_csv(path, sep=sep)
        rep["StudyInstanceUID"] = rep["StudyInstanceUID"].astype(str)
        rep = rep.set_index("StudyInstanceUID")[LAB]
        rep = rep.loc[rep.index.intersection(ids)]
        tf, tco = aucs(fa_o.loc[ids], rep), aucs(co_o.loc[ids], rep)
        print(f"\n(b) OOF vs {name} report labels (n={len(rep)})")
        print(f"  FamilyA {summ(tf)}\n  CoAt    {summ(tco)}")
        for w in (0.3, 0.45):
            print(f"  blend{int(w * 100):02d} {summ(aucs(rank_blend(fa_o.loc[ids], co_o.loc[ids], w), rep))}")
        res["b"][name] = {"fa": tf, "coat": tco}

    # (c) rank correlation
    print("\n(c) Spearman vs CoAt OOF / public DINO OOF (per finding)")
    res["c"] = {}
    di = ids.intersection(dino.index)
    for f in LAB:
        rc = spearmanr(fa_o.loc[ids, f], co_o.loc[ids, f]).statistic if len(ids) > 2 else np.nan
        rd = spearmanr(fa_o.loc[di, f], dino.loc[di, f]).statistic if len(di) > 2 else np.nan
        rcd = spearmanr(co_o.loc[di, f], dino.loc[di, f]).statistic if len(di) > 2 else np.nan
        res["c"][f] = {"fa_coat": float(rc), "fa_dino": float(rd), "coat_dino": float(rcd)}
        mark = " *" if f in WEAK5 else ""
        print(f"  {f:17s} FA-CoAt {rc:.3f} | FA-DINO {rd:.3f} | CoAt-DINO {rcd:.3f}{mark}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
