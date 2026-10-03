#!/usr/bin/env python3
"""Pseudo-label round: report labels + CLEAN out-of-fold image predictions -> next round's targets.

    python scripts/make_round_labels.py --out labels_r1.parquet \
        --family 'outputs/runpod_results/n-5bc0a0f45d75/*,outputs/runpod_results/r0-coat-f*/*' \
        --family 'outputs/runpod_results/fa-f0-rep/*,outputs/runpod_results/r0-fa-f*/*'

Why (PLAN_TO_TOP 2026-10-03 evening): on holdout-29 the report labels agree with the experts at 0.883,
a clean OOF ensemble at 0.929, their blend at 0.941 — the images know more than the reports, mostly where
the report is silent (pilk verdict UNK: Synovitis 84 %, Fracture 56 %, Baker's 46 %, Lateral OA 33 %).

Teachers must be CLEAN: ImageNet-init models predicting studies of a fold they never trained on. Runs
initialised from public v5 (trained on all studies) only echo the report labels and are refused.

Per finding:  e = mean over families of the within-family mean rank of the OOF predictions;
              p = Platt(e -> report label)            (probability scale of the report labels)
              target = a * report + (1 - a) * p,      a = --a-said (report says YES/NO) or --a-unk (silent)
Each --family is one glob of result dirs (all folds of one model family; several seeds are averaged).
Expert studies are left to the trainers (gold labels / holdout). Prints holdout-29 AUC of report, p, target.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PILK = REPO / "data" / "external" / "public_labels" / "pilk" / "report_labels_v2.csv"


def _frames(pattern: str, name: str) -> list[pd.DataFrame]:
    out = []
    for d in sorted(d for pat in pattern.split(",") for d in glob.glob(pat)):   # a family may span several globs
        for j in glob.glob(f"{d}/raptor_ft_*.json"):
            if "init:" in str(json.loads(Path(j).read_text()).get("src", "")):
                raise SystemExit(f"{j}: initialised from another checkpoint (leaky teacher) — refuse")
        for f in sorted(glob.glob(f"{d}/{name}")):
            if re.search(r"_(swa|ema|last|best|epens|ep\d+)(_fold\d+)?\.npz$", f):
                continue
            z = np.load(f, allow_pickle=True)
            out.append(pd.DataFrame(z["pred"], index=z["ids"].astype(str), columns=LAB))
    return out


def family_rank(pattern: str, name: str) -> pd.DataFrame:
    """Rank within each model's own predictions (fold models differ in scale), then mean per study."""
    fr = [f.rank(pct=True) for f in _frames(pattern, name)]
    if not fr:
        return pd.DataFrame(columns=LAB)
    return pd.concat(fr).groupby(level=0).mean()


def auc_row(P: pd.DataFrame, Y: pd.DataFrame) -> dict[str, float]:
    ids = P.index.intersection(Y.index)
    out = {}
    for f in LAB:
        y = (Y.loc[ids, f] >= 0.5).astype(int)
        if 0 < y.sum() < len(y) and P.loc[ids, f].notna().all():
            out[f] = float(roc_auc_score(y, P.loc[ids, f]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", action="append", required=True, help="glob of result dirs of one family")
    ap.add_argument("--report", default="labels_bestof_v1.parquet", help="report-derived soft labels (data/processed)")
    ap.add_argument("--out", required=True, help="parquet name under data/processed")
    ap.add_argument("--a-said", type=float, default=0.5, help="report weight where the report addresses the finding")
    ap.add_argument("--a-unk", type=float, default=0.2, help="report weight where the report is silent (pilk UNK)")
    ap.add_argument("--min-cover", type=float, default=0.98, help="required OOF coverage of non-expert studies")
    a = ap.parse_args()

    rep = pd.read_parquet(REPO / "data" / "processed" / a.report)
    rep["StudyInstanceUID"] = rep["StudyInstanceUID"].astype(str)
    rep = rep.set_index("StudyInstanceUID")[LAB].astype(float)
    pk = pd.read_csv(PILK)
    pk["StudyInstanceUID"] = pk["StudyInstanceUID"].astype(str)
    pk = pk.set_index("StudyInstanceUID")
    tr = pd.read_csv(REPO / "data" / "raw" / "train.csv")
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    gold = tr.set_index("StudyInstanceUID")[LAB].dropna()
    hold = [u for u in json.loads((REPO / "agent" / "protocol" / "expert_holdout.json").read_text()) if u in gold.index]

    oof = [family_rank(p, "raptor_oof_*_fold[0-9].npz") for p in a.family]
    hol = [family_rank(p, "raptor_holdout_*.npz") for p in a.family]
    for p, o in zip(a.family, oof):
        print(f"family {p}: OOF studies {len(o)}")
    ids = sorted(set.intersection(*(set(o.index) for o in oof)) & set(rep.index) - set(gold.index))
    nonexp = [u for u in rep.index if u not in gold.index]
    cover = len(ids) / len(nonexp)
    print(f"OOF coverage {len(ids)}/{len(nonexp)} = {cover:.3f}")
    E = sum(o.loc[ids] for o in oof) / len(oof)
    hid = sorted(set.intersection(*(set(h.index) for h in hol)) & set(hold)) if all(len(h) for h in hol) else []
    Eh = sum(h.loc[hid] for h in hol) / len(hol) if hid else None

    out = rep.copy()
    P = pd.DataFrame(index=ids, columns=LAB, dtype=float)
    Ph = pd.DataFrame(index=hid, columns=LAB, dtype=float)
    print(f"\n{'finding':17s} silent%  prev(report) prev(target)  mean|target-report|")
    for f in LAB:
        y = (rep.loc[ids, f] >= 0.5).astype(int)
        lr = LogisticRegression(C=10.0).fit(E[[f]].values, y)
        P[f] = lr.predict_proba(E[[f]].values)[:, 1]
        if hid:   # holdout ranks are within 29 studies: same 0..1 scale as E
            Ph[f] = lr.predict_proba(Eh[[f]].values)[:, 1]
        silent = (pk[f + "__verdict"].reindex(ids) == "UNK").fillna(False).values
        w = np.where(silent, a.a_unk, a.a_said)
        tgt = w * rep.loc[ids, f].values + (1 - w) * P[f].values
        print(f"{f:17s} {silent.mean():6.2f}  {y.mean():10.3f} {(tgt >= 0.5).mean():11.3f}  "
              f"{np.abs(tgt - rep.loc[ids, f].values).mean():10.3f}")
        out.loc[ids, f] = tgt

    if hid:
        rh = pk.loc[hid, LAB].astype(float)
        sil = pd.DataFrame({f: (pk.loc[hid, f + "__verdict"] == "UNK") for f in LAB})
        wh = np.where(sil.values, a.a_unk, a.a_said)
        th = pd.DataFrame(wh * rh.values + (1 - wh) * Ph.values, index=hid, columns=LAB)
        print(f"\nholdout-{len(hid)} AUC vs experts (macro over scorable findings)")
        for name, frame in (("report (pilk)", rh), ("clean OOF ensemble", Ph), ("round target", th)):
            t = auc_row(frame, gold.loc[hid])
            print(f"  {name:20s} {np.mean(list(t.values())):.4f}  " + " ".join(f"{v:.2f}" for v in t.values()))
    if cover < a.min_cover:
        raise SystemExit(f"coverage {cover:.3f} < {a.min_cover}: not all folds present — labels NOT written")
    dest = REPO / "data" / "processed" / a.out
    out.reset_index().to_parquet(dest, index=False)
    print(f"\nwrote {dest} ({len(out)} studies; {len(ids)} re-targeted)")


if __name__ == "__main__":
    main()
