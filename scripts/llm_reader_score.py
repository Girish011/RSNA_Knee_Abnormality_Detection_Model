#!/usr/bin/env python3
"""Score the LLM reader's blind answers on the expert-labelled warm-up studies (9 Oct).

    python scripts/llm_reader_score.py --bar                 # our own models on the 20 warm-up studies (the bar)
    python scripts/llm_reader_score.py --packets <dir>       # + the reader's answer.json files found under <dir>

Reader confidence = its probability that the experts mark the finding present. Reported on the studies read so far:
macro AUC over the findings that have both classes there (reader, own models, rank mean of the two), and the
Present / Absent agreement (Unsure counts as a miss). Own models = the four first-round ConvNeXt-S (0.937 alone on the
leaderboard) unless --models is given. Nothing here is a leaderboard score.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import expert_ruler as er   # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DL = REPO / "outputs" / "doctor_label"
LAB = er.LAB
OWN = ",".join(f"r2-cnxs336-f{k}:r2cnxs336{k}" for k in range(4))


def macro(y: pd.DataFrame, p: pd.DataFrame) -> tuple[float, dict]:
    per = {k: roc_auc_score(y[k], p[k]) for k in LAB if 0 < y[k].sum() < len(y)}
    return (float(np.mean(list(per.values()))) if per else float("nan")), per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packets", type=Path)
    ap.add_argument("--bar", action="store_true")
    ap.add_argument("--models", default=OWN)
    ap.add_argument("--skip", default="", help="comma list of packet ids left out (the pilot)")
    a = ap.parse_args()
    cases = {c["case"]: c["uid"] for c in json.loads((DL / "cases.json").read_text()) if c["set"] == "warmup"}
    gold = pd.read_csv(REPO / "data" / "raw" / "train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    P = [er.expert_preds(s) for s in a.models.split(",")]
    own = sum(p.loc[list(cases.values()), LAB].apply(rankdata) for p in P) / len(P)
    own.index = list(cases.values())
    Y = gold.loc[list(cases.values()), LAB].astype(int)
    m, per = macro(Y, own)
    print(f"BAR: own models ({len(P)}) on the {len(Y)} warm-up studies: macro AUC {m:.4f} over {len(per)} findings (offline)")
    if not a.packets:
        return
    mp = json.loads((DL / "claude_reader" / "map_warmup.json").read_text())["map"]
    rows, ans = {}, {}
    for f in sorted(a.packets.glob("*/answer.json")):
        if f.parent.name in a.skip.split(","):
            continue
        d = json.loads(f.read_text())
        uid = cases[mp[f.parent.name]]
        rows[uid] = {k: float(d["findings"][k]["confidence"]) for k in LAB}
        ans[uid] = {k: d["findings"][k]["answer"] for k in LAB}
    if not rows:
        print("no answer.json yet"); return
    C = pd.DataFrame(rows).T[LAB]; A = pd.DataFrame(ans).T[LAB]; y = Y.loc[C.index]; o = own.loc[C.index]
    agree = float(np.mean([(A[k] == np.where(y[k] == 1, "Present", "Absent")).mean() for k in LAB]))
    print(f"\nreader: {len(C)} studies | Present / Absent agreement with the experts {agree:.3f} "
          f"| expert positives {int(y.values.sum())} of {y.size} cells | reader Present {int((A == 'Present').values.sum())}, Unsure {int((A == 'Unsure').values.sum())}")
    if len(C) >= 6:
        both = C.apply(rankdata) / len(C) + o.apply(rankdata) / len(C)
        for name, p in (("reader", C), ("own models", o), ("reader + own (rank mean)", both)):
            mm, pp = macro(y, p)
            print(f"  {name:26s} macro AUC {mm:.4f} over {len(pp)} findings")
    else:   # too few studies for per-finding AUCs: pooled over all cells instead
        print(f"  pooled over the {y.size} cells: reader AUC {roc_auc_score(y.values.ravel(), C.values.ravel()):.3f} "
              f"| own models AUC {roc_auc_score(y.values.ravel(), (o.apply(rankdata) / len(o)).values.ravel()):.3f} (ranks within these studies; rough)")
    print("\nper study (expert positives | reader Present | missed | extra):")
    for uid in C.index:
        e = {k for k in LAB if y.loc[uid, k] == 1}; r = {k for k in LAB if A.loc[uid, k] == "Present"}
        print(f"  {[k for k, v in mp.items() if cases[v] == uid][0]}: {len(e)} | {len(r)} | missed {sorted(e - r)} | extra {sorted(r - e)}")


if __name__ == "__main__":
    main()
