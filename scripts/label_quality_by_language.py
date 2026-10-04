#!/usr/bin/env python3
"""Label quality by report language (no GPU): where are the report labels weakest, and where did a round move them?

    python scripts/label_quality_by_language.py [--round-labels labels_r1.parquet] \
        --family 'outputs/runpod_results/n-5bc0a0f45d75/*,outputs/runpod_results/r0-coat-f*/*' \
        --family 'outputs/runpod_results/fa-f0-rep/*,outputs/runpod_results/r0-fa-f*/*'

Per language (stop-word vote; Greek/Cyrillic/Turkish by script) over the non-expert studies:
  silent      share of finding cells the report does not address (pilk verdict UNK)
  readers     AUC of one report reader (dread, soft) against the other (pilk YES/NO), report-addressed cells only —
              two independent LLM readers of the same text; low = the text is read unreliably in this language
  img~rep     macro AUC of the clean OOF teacher ensemble against the report labels; low = images and reports
              disagree more (report noise, or harder / different images — not separable here)
  moved       mean |round label - report label|
  vs experts  macro AUC of the pilk report labels on the expert studies of that language (all 58; tiny n)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from make_round_labels import LAB, PILK, family_rank  # noqa: E402

STOP = {
    "fr": "le la les des du une est et avec sans pas de au aux dans sur genou ménisque rupture épanchement".split(),
    "es": "el la los las del una es y con sin no de en se rodilla menisco rotura derrame hallazgos".split(),
    "en": "the of and is are with without no in there knee meniscus tear effusion seen normal".split(),
    "pt": "o a os as do da uma é e com sem não de em joelho menisco rotura derrame são".split(),
    "it": "il la lo gli del della una è e con senza non di in ginocchio menisco lesione versamento".split(),
    "nl": "het een van geen niet met zonder en is de knie meniscus scheur vocht normaal aspect bij ter".split(),
    "hr": "je se uz na u i bez nema uredan uredna uredne koljena meniska nalazi signala izljeva lijevog desnog".split(),
    "de": "der die das des ein eine ist und mit ohne kein nicht im knie meniskus riss erguss".split(),
}


def language(text: str) -> str:
    t = str(text or "")
    if re.search(r"[Ͱ-Ͽἀ-῿]", t):
        return "el"
    if len(re.findall(r"[\u0400-\u04ff]", t)) > 20:
        return "bg"   # Cyrillic reports in this corpus are Bulgarian
    if re.search(r"[şğİı]", t) or re.search(r"\b(diz|normaldir|yoktur|izlen\w*|eklem|kemik)\b", t, re.I):
        return "tr"
    words = re.findall(r"[^\W\d_]+", t.lower())
    score = {k: sum(w in set(v) for w in words) for k, v in STOP.items()}
    best = max(score, key=score.get)
    return best if score[best] >= 3 else "other"


def macro_auc(P: pd.DataFrame, Y: pd.DataFrame, mask: pd.DataFrame | None = None, min_pos: int = 5) -> tuple[float, int]:
    out = []
    for f in LAB:
        m = mask[f].values if mask is not None else np.ones(len(P), bool)
        y = (Y[f].values[m] >= 0.5).astype(int)
        if min(y.sum(), len(y) - y.sum()) >= min_pos:
            out.append(roc_auc_score(y, P[f].values[m]))
    return (float(np.mean(out)) if out else float("nan")), len(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", action="append", required=True)
    ap.add_argument("--report", default="labels_bestof_v1.parquet")
    ap.add_argument("--round-labels", default="labels_r1.parquet")
    a = ap.parse_args()

    tr = pd.read_csv(REPO / "data" / "raw" / "train.csv")
    tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
    tr = tr.set_index("StudyInstanceUID")
    lang = tr["Report"].map(language)
    gold = tr[LAB].dropna()

    def load(p: Path) -> pd.DataFrame:
        d = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        d["StudyInstanceUID"] = d["StudyInstanceUID"].astype(str)
        return d.set_index("StudyInstanceUID")

    rep = load(REPO / "data" / "processed" / a.report)[LAB].astype(float)
    new = load(REPO / "data" / "processed" / a.round_labels)[LAB].astype(float)
    pk = load(PILK)
    dread = load(REPO / "data" / "external" / "public_labels" / "dread" / "labels_llm_soft.csv")[LAB].astype(float)
    oof = [family_rank(p, "raptor_oof_*_fold[0-9].npz") for p in a.family]
    ids = sorted(set.intersection(*(set(o.index) for o in oof)) & set(rep.index) - set(gold.index))
    E = sum(o.loc[ids] for o in oof) / len(oof)
    said = pd.DataFrame({f: (pk[f + "__verdict"].reindex(ids) != "UNK") for f in LAB}).fillna(False)

    print(f"{'lang':6s} {'studies':>7s} {'silent':>7s} {'readers':>8s} {'img~rep':>8s} {'moved':>6s} "
          f"{'experts':>8s} {'vs experts':>11s}")
    rows = []
    for lg, n in lang.loc[ids].value_counts().items():
        u = [i for i in ids if lang[i] == lg]
        g = [i for i in gold.index if lang[i] == lg and i in pk.index]
        readers, _ = macro_auc(dread.reindex(u), pk.loc[u, LAB].astype(float), said.loc[u])
        img, k = macro_auc(E.loc[u], rep.loc[u])
        exp, ke = macro_auc(pk.loc[g, LAB].astype(float), gold.loc[g], min_pos=2) if g else (float("nan"), 0)
        rows.append((lg, n, 1 - said.loc[u].values.mean(), readers, img, k,
                     np.abs(new.loc[u] - rep.loc[u]).values.mean(), len(g), exp, ke))
        print(f"{lg:6s} {n:7d} {rows[-1][2]:7.2f} {readers:8.3f} {img:8.3f} {rows[-1][6]:6.3f} "
              f"{len(g):8d} {exp:8.3f} ({ke} f)   [img~rep over {k} findings]")

    print("\nper finding: mean |round - report| by language")
    top = [r[0] for r in rows if r[1] >= 100]
    print(f"{'finding':17s} " + " ".join(f"{t:>6s}" for t in top))
    for f in LAB:
        print(f"{f:17s} " + " ".join(
            f"{np.abs(new.loc[[i for i in ids if lang[i] == t], f] - rep.loc[[i for i in ids if lang[i] == t], f]).mean():6.3f}"
            for t in top))
    print("\nper finding: report silent share by language")
    for f in LAB:
        print(f"{f:17s} " + " ".join(f"{1 - said.loc[[i for i in ids if lang[i] == t], f].mean():6.2f}" for t in top))


if __name__ == "__main__":
    main()
