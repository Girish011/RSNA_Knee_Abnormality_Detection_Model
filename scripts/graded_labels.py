#!/usr/bin/env python3
"""Severity-graded report labels (train-only LLM pass) and their agreement with the 58 experts.

    python scripts/graded_labels.py --experts                 # the 58 expert studies only (cents), prints the AUCs
    python scripts/graded_labels.py --all [--workers 16]      # every train report -> data/processed/labels_graded_v1.parquet

Why (6 Oct): report YES is a superset of expert YES in almost every finding (57 experts: ACL 23 of 32 report-YES
confirmed, MCL 9 / 18, Effusion 34 / 50, Contusion 16 / 36, Lateral OA 9 / 16), report NO / silent is reliable. The
label files so far hold a presence probability (hedging-based), which gives "small effusion" and "large effusion" the
same target. This pass asks for a grade per finding on a fixed scale written before looking at any expert label:
  0 explicitly normal / absent   1 not mentioned   2 minimal, doubtful or low-grade
  3 definite, moderate           4 severe / unequivocal / complete
Responses are cached in agent/labels/cache_graded/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rsna_knee.agent.deepseek import FLASH, PRO, chat_json

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO / "agent" / "labels" / "cache_graded"
VERSION = "g1"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SCALE = {
    "ACL": "2 = sprain, mucoid degeneration, thickening or low-grade partial tear; 3 = high-grade partial tear or probable tear; 4 = complete tear / rupture, or torn graft",
    "MCL": "2 = grade 1 sprain, periligamentous edema; 3 = grade 2 / partial tear; 4 = grade 3 / complete tear",
    "Medial Meniscus": "2 = intrasubstance signal / degeneration without surface tear, fraying, doubtful tear; 3 = definite tear; 4 = complex, displaced, bucket-handle or root tear, or large tear with extrusion",
    "Lateral Meniscus": "2 = intrasubstance signal / degeneration without surface tear, fraying, doubtful tear; 3 = definite tear; 4 = complex, displaced, bucket-handle or root tear, or large tear with extrusion",
    "Medial OA": "medial femorotibial compartment only. 2 = mild / grade 1-2 chondropathy, incipient degenerative change; 3 = moderate, grade 3 cartilage loss, osteophytes; 4 = severe, full-thickness loss, bone-on-bone, subchondral changes",
    "Lateral OA": "lateral femorotibial compartment only. 2 = mild / grade 1-2 chondropathy, incipient degenerative change; 3 = moderate, grade 3 cartilage loss, osteophytes; 4 = severe, full-thickness loss, subchondral changes. A generic 'degenerative changes' or 'tricompartmental' statement without a lateral-specific description is 2",
    "PF OA": "patellofemoral compartment only. 2 = mild / grade 1-2 chondromalacia patellae; 3 = moderate, grade 3; 4 = severe, full-thickness, osteophytes / subchondral changes",
    "Effusion": "2 = small, mild, minimal or physiologic amount of fluid; 3 = moderate; 4 = large / marked / tense, or lipohemarthrosis",
    "Synovitis": "2 = mild synovial thickening, or only implied by effusion wording; 3 = definite synovitis / synovial hypertrophy; 4 = marked, nodular or proliferative synovitis",
    "Baker's": "2 = small / minimal popliteal cyst or fluid in the gastrocnemius-semimembranosus bursa; 3 = definite cyst; 4 = large, septated or ruptured cyst",
    "Contusion": "bone marrow edema / contusion. 2 = small, faint or subchondral edema of degenerative type; 3 = definite contusion / bone bruise; 4 = extensive, multiple or kissing contusions",
    "Fracture": "2 = doubtful, old / healed, or small osteochondral irregularity; 3 = definite non-displaced, subchondral insufficiency or avulsion fracture; 4 = displaced, depressed or comminuted fracture",
}
SYSTEM = ("You are a musculoskeletal radiologist reading a knee MRI report that may be in any language. Grade each "
          "requested finding on this scale, using only the report: 0 = explicitly normal / absent; 1 = not mentioned; "
          "2 = minimal, doubtful or low-grade; 3 = definite, moderate; 4 = severe, unequivocal or complete. "
          "The meaning of 2, 3 and 4 is given per finding. Grade the knee as described, do not infer findings the "
          'report does not describe. Reply as JSON {"<finding>": <integer 0-4>, ...} with every requested finding.')


def read_one(uid: str, report: str, model: str) -> tuple[dict, float]:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (hashlib.sha1(f"{VERSION}|{model}|{uid}".encode()).hexdigest()[:20] + ".json")
    if f.exists():
        return json.loads(f.read_text()), 0.0
    user = "Findings:\n" + "\n".join(f"- {l}: {SCALE[l]}" for l in LAB) + f"\n\nReport:\n{report}"
    out, cost = chat_json(SYSTEM, user, model=model)
    g = {l: int(np.clip(int(round(float(out[l]))), 0, 4)) for l in LAB if l in out}
    if len(g) == len(LAB):
        f.write_text(json.dumps(g))
    return g, cost


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default="flash", choices=["flash", "pro"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="labels_graded_v1.parquet")
    ap.add_argument("--max-usd", type=float, default=14.0,
                    help="stop sending new reports once this run's estimated cost passes it (cached reads are free; "
                         "rerun to continue). The harness cap for relabels is $15 a day")
    a = ap.parse_args()
    model = {"flash": FLASH, "pro": PRO}[a.model]
    tr = pd.read_csv(REPO / "data/raw/train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    gold = tr.dropna(subset=LAB)
    ids = list(gold.index) if a.experts else list(tr.index)

    def job(u):
        try:
            return u, *read_one(u, str(tr.loc[u, "Report"]), model)
        except Exception as e:  # noqa: BLE001
            return u, {"err": f"{type(e).__name__}: {e}"}, 0.0

    rows, cost, bad, done = {}, 0.0, 0, 0
    with ThreadPoolExecutor(a.workers) as ex:
        for k in range(0, len(ids), 200):          # chunks, so that the cost guard can stop the run
            if cost >= a.max_usd:
                print(f"cost guard: ${cost:.2f} >= ${a.max_usd:.2f}; {len(ids) - done} reports left for the next run", flush=True)
                break
            for u, g, c in ex.map(job, ids[k:k + 200]):
                cost += c; done += 1
                if len(g) == len(LAB):
                    rows[u] = g
                else:
                    bad += 1
            print(f"{done}/{len(ids)} | ${cost:.2f} | failed {bad}", flush=True)
    G = pd.DataFrame(rows).T[LAB].astype(int)
    print(f"graded {len(G)}/{len(ids)} reports | ${cost:.3f} | failed {bad}")
    if a.all:
        G.reset_index(names="StudyInstanceUID").to_parquet(REPO / "data/processed" / a.out, index=False)
        print("wrote", a.out, "| grade counts:\n", G.apply(lambda c: c.value_counts()).fillna(0).astype(int).T.to_string())
    ex_ids = [u for u in gold.index if u in G.index]
    pk = pd.read_csv(REPO / "data/external/public_labels/pilk/report_labels_v2.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    both = [u for u in ex_ids if u in pk.index]
    Y = gold.loc[both, LAB].astype(int)
    out = []
    for f in LAB:
        # grade as an ordinal score; 1 (not mentioned) sits between 0 and 2 by construction
        out.append({"finding": f, "pilk prob": roc_auc_score(Y[f], pk.loc[both, f].astype(float)),
                    "graded": roc_auc_score(Y[f], G.loc[both, f]),
                    "graded + 0.01 x pilk": roc_auc_score(Y[f], G.loc[both, f] + 0.01 * pk.loc[both, f].astype(float)),
                    "P(expert+) by grade 0..4": " ".join(f"{int(Y[f][G.loc[both, f] == k].sum())}/{int((G.loc[both, f] == k).sum())}" for k in range(5))})
    T = pd.DataFrame(out).set_index("finding")
    print(f"\n{len(both)} experts with both label sources: AUC of the report label vs the expert label")
    print(T.round(3).to_string())
    print("macro:", {c: round(float(T[c].mean()), 4) for c in T.columns[:3]})


if __name__ == "__main__":
    main()
