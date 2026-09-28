"""LLM re-reading of weak label cells (train-only), audited on the expert-dev split.

Only cells the public readers are unsure about are re-read: pilk verdict UNK, or pilk and dread
disagreeing by more than ``DISAGREE``. The same reader is run on the 29 expert-dev studies for
the requested findings; a finding is kept only if the reader is not worse than pilk there and
its precision at 0.5 is >= 0.5. The expert holdout is never read here. Replaced cells become
mean(base, reader). Responses are cached on disk, so a repeated request costs nothing.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, roc_auc_score

from rsna_knee.agent import genome as G
from rsna_knee.agent.deepseek import FLASH, PRICE, PRO, chat_json
from rsna_knee.constants import LABEL_COLS

REPO = G.REPO
PUB = REPO / "data" / "external" / "public_labels"
CACHE = REPO / "agent" / "labels" / "cache"
PROMPT_VERSION = "v1"
DISAGREE = 0.35
EST_USD_PER_CALL = 0.0007  # flash, measured 2026-09-24 relabels; pro scales by PRICE ratio
RULES = ("pilk_unk", "disagree", "both", "all")  # all = re-read every non-expert study
MODELS = {"flash": FLASH, "pro": PRO}

DEFS = {
    "ACL": "anterior cruciate ligament tear (partial or complete), including graft tear",
    "MCL": "medial collateral ligament tear or sprain grade 2-3",
    "Medial Meniscus": "tear of the medial meniscus (any horn/type)",
    "Lateral Meniscus": "tear of the lateral meniscus (any horn/type)",
    "Medial OA": "osteoarthritis / chondrosis of the medial femorotibial compartment",
    "Lateral OA": "osteoarthritis / chondrosis of the lateral femorotibial compartment",
    "PF OA": "patellofemoral osteoarthritis / chondromalacia patellae",
    "Effusion": "joint effusion (more than physiologic fluid)",
    "Synovitis": "synovitis or synovial thickening/hypertrophy",
    "Baker's": "Baker's (popliteal) cyst",
    "Contusion": "bone contusion / bone marrow edema / bone bruise",
    "Fracture": "fracture, including osteochondral or subchondral insufficiency fracture",
}
SYSTEM = ("You are a musculoskeletal radiologist reading a knee MRI report that may be in any "
          "language. For each requested finding give the probability (0-1) that the finding is "
          "present in this knee, based only on the report. If the report does not mention a "
          "finding, estimate it from what an unremarkable mention implies (usually low, 0.05-0.2). "
          "Hedged wording ('possible', 'cannot exclude') is 0.4-0.7. Reply as JSON {finding: probability}.")


def _key(uid: str, labels: list[str], model: str) -> str:
    return hashlib.sha1(f"{PROMPT_VERSION}|{model}|{uid}|{','.join(labels)}".encode()).hexdigest()[:20]


def read_one(uid: str, report: str, labels: list[str], model: str = FLASH, llm=chat_json) -> tuple[dict, float]:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{_key(uid, labels, model)}.json"
    if f.exists():
        return json.loads(f.read_text()), 0.0
    user = "Findings:\n" + "\n".join(f"- {l}: {DEFS[l]}" for l in labels) + f"\n\nReport:\n{report}"
    out, cost = llm(SYSTEM, user, model=model)
    probs = {l: float(np.clip(float(out.get(l, np.nan)), 0, 1)) for l in labels if l in out}
    f.write_text(json.dumps(probs))
    return probs, cost


def select_cells(labels: list[str], rule: str) -> dict[str, list[str]]:
    """Non-expert study -> findings to re-read."""
    pilk = pd.read_csv(PUB / "pilk" / "report_labels_v2.csv").set_index("StudyInstanceUID")
    dread = pd.read_csv(PUB / "dread" / "labels_llm_soft.csv").set_index("StudyInstanceUID")
    ids = pilk.index.intersection(dread.index)
    cells: dict[str, list[str]] = {}
    for l in labels:
        unk = pilk.loc[ids, f"{l}__verdict"] == "UNK"
        dis = (pilk.loc[ids, l] - dread.loc[ids, l]).abs() > DISAGREE
        if rule == "all":
            m = pd.Series(True, index=ids)
        else:
            m = unk if rule == "pilk_unk" else dis if rule == "disagree" else (unk | dis)
        for uid in ids[m.to_numpy()]:
            cells.setdefault(uid, []).append(l)
    return cells


def estimate(labels: list[str], rule: str, model: str = FLASH) -> tuple[int, float]:
    n = len(select_cells(labels, rule)) + 29
    scale = PRICE.get(model, PRICE[PRO])[1] / PRICE[FLASH][1]
    return n, n * EST_USD_PER_CALL * scale


def run(labels: list[str], rule: str, base: str, *, model: str = FLASH, workers: int = 16,
        llm=chat_json) -> dict:
    train = pd.read_csv(REPO / "data" / "raw" / "train.csv").set_index("StudyInstanceUID")
    split = json.loads((REPO / "agent" / "protocol" / "expert_split.json").read_text())
    dev = split["dev"]
    cells = select_cells(labels, rule)
    jobs = [(u, labels) for u in dev] + list(cells.items())
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda j: read_one(j[0], str(train.loc[j[0], "Report"]), sorted(j[1]), model, llm), jobs))
    cost = float(sum(c for _, c in results))
    reads = {u: r for (u, _), (r, _) in zip(jobs, results)}

    pilk = pd.read_csv(PUB / "pilk" / "report_labels_v2.csv").set_index("StudyInstanceUID")
    audit, kept = {}, []
    dev_p = np.array([u for u in dev if u in pilk.index])
    for l in labels:
        y = train.loc[dev_p, l].astype(int).to_numpy()
        s = np.array([reads[u].get(l, np.nan) for u in dev_p])
        ok = ~np.isnan(s)
        if ok.sum() < 10 or len(set(y[ok])) < 2:
            audit[l] = {"kept": False, "reason": "too few dev reads"}
            continue
        auc = roc_auc_score(y[ok], s[ok])
        pauc = roc_auc_score(y[ok], pilk.loc[dev_p[ok], l].to_numpy())
        prec = precision_score(y[ok], (s[ok] > 0.5).astype(int), zero_division=0)
        keep = bool(auc >= pauc - 0.01 and prec >= 0.5)
        audit[l] = {"dev_auc": round(auc, 3), "pilk_dev_auc": round(pauc, 3), "prec": round(prec, 2), "kept": keep}
        if keep:
            kept.append(l)

    reg = G.label_registry()
    soft = pd.read_parquet(REPO / "data" / "processed" / reg[base]["file"]).set_index("StudyInstanceUID")
    n_changed = 0
    for uid, ls in cells.items():
        for l in ls:
            if l in kept and uid in soft.index and l in reads.get(uid, {}):
                soft.loc[uid, l] = 0.5 * soft.loc[uid, l] + 0.5 * reads[uid][l]
                n_changed += 1
    name = f"relabel_{hashlib.sha1(json.dumps([sorted(labels), rule, base, PROMPT_VERSION, model]).encode()).hexdigest()[:8]}"
    out = {"name": name, "labels": labels, "rule": rule, "base": base, "audit": audit, "kept": kept,
           "cells_read": sum(len(v) for v in cells.values()), "cells_changed": n_changed,
           "cost_usd": round(cost, 3), "registered": bool(kept)}
    if kept:
        fname = f"labels_{name}.parquet"
        soft.reset_index().to_parquet(REPO / "data" / "processed" / fname, index=False)
        G.LABEL_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(G.LABEL_REGISTRY.read_text()) if G.LABEL_REGISTRY.exists() else {}
        existing[name] = {"file": fname, "audit": {l: audit[l] for l in kept}, "base": base, "rule": rule}
        G.LABEL_REGISTRY.write_text(json.dumps(existing, indent=1))
    return out
