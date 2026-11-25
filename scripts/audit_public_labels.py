#!/usr/bin/env python3
"""Audit report-label sources vs the expert-labeled studies before any train.

Sources: public `pilkwang/rsna-knee-llm-labels` (report_labels_v2.csv), public
`dreaddevelopment/rsna-knee-labels` (labels_llm_soft.csv, ships without the expert studies),
and our keyword extractors v1 / v7 run on the raw reports (no expert override).

Per label on the experts: AUC of the soft score, precision / recall at 0.5, and the plan's
entry gate (precision >= 0.5). Macro AUC gets a study-bootstrap CI. Sources that cannot be
graded on the experts are compared to the others on the non-expert studies instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from rsna_knee.constants import LABEL_COLS
from rsna_knee.text.weak_labels_v1 import weak_labels_for_report as v1_for_report
from rsna_knee.text.weak_labels_v7 import weak_labels_v7_for_report as v7_for_report

PUB = Path("data/external/public_labels")
OUT = Path("docs/audit")
PREC_GATE = 0.5
N_BOOT = 2000


def keyword_soft(train: pd.DataFrame, fn) -> pd.DataFrame:
    """Signed confidence: positive -> 0.5+conf/2, negative -> 0.5-conf/2, silent -> 0.5."""
    rows = []
    for report in train["Report"].astype(str):
        out = fn(report)
        row = {}
        for c in LABEL_COLS:
            v, conf = out[c], float(out[f"{c}__conf"])
            if v is None or conf == 0.0 or pd.isna(v):
                row[c] = 0.5
            else:
                row[c] = 0.5 + conf / 2 if float(v) == 1.0 else 0.5 - conf / 2
        rows.append(row)
    return pd.DataFrame(rows).assign(StudyInstanceUID=train["StudyInstanceUID"].to_numpy())


def load_sources(train: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pilk = pd.read_csv(PUB / "pilk" / "report_labels_v2.csv")
    dread = pd.read_csv(PUB / "dread" / "labels_llm_soft.csv")
    return {
        "pilk": pilk[["StudyInstanceUID", *LABEL_COLS]],
        "dread": dread[["StudyInstanceUID", *LABEL_COLS]],
        "kw_v1": keyword_soft(train, v1_for_report),
        "kw_v7": keyword_soft(train, v7_for_report),
    }


def grade(y: pd.DataFrame, s: pd.DataFrame) -> list[dict]:
    rows = []
    for c in LABEL_COLS:
        yt, ys = y[c].astype(int).to_numpy(), s[c].astype(float).to_numpy()
        pred = (ys > 0.5).astype(int)
        prec = precision_score(yt, pred, zero_division=0)
        rows.append({
            "label": c, "n": len(yt), "expert_pos": int(yt.sum()), "pred_pos": int(pred.sum()),
            "auc": roc_auc_score(yt, ys), "prec": prec, "rec": recall_score(yt, pred, zero_division=0),
            "gate_pass": bool(pred.sum() > 0 and prec >= PREC_GATE),
        })
    return rows


def boot_macro(y: pd.DataFrame, s: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    yt, ys = y[LABEL_COLS].to_numpy(int), s[LABEL_COLS].to_numpy(float)
    out = []
    while len(out) < N_BOOT:
        idx = rng.integers(0, len(yt), len(yt))
        aucs = [roc_auc_score(yt[idx, j], ys[idx, j]) for j in range(len(LABEL_COLS))
                if 0 < yt[idx, j].sum() < len(idx)]
        if len(aucs) == len(LABEL_COLS):
            out.append(np.mean(aucs))
    return np.asarray(out)


def main() -> None:
    train = pd.read_csv("data/raw/train.csv")
    experts = train.loc[train[LABEL_COLS].notna().all(axis=1), ["StudyInstanceUID", *LABEL_COLS]]
    sources = load_sources(train)
    rng = np.random.default_rng(0)

    graded, per_label, boots = {}, [], {}
    common = set(experts["StudyInstanceUID"])
    for s in sources.values():
        ids = set(s["StudyInstanceUID"])
        if ids & common:
            common &= ids
    common = sorted(common)
    y = experts.set_index("StudyInstanceUID").loc[common]

    for name, s in sources.items():
        s = s.set_index("StudyInstanceUID")
        if not set(common) <= set(s.index):
            graded[name] = {"graded": False, "reason": "no expert studies in source"}
            continue
        sx = s.loc[common]
        rows = grade(y, sx)
        per_label += [{"source": name, **r} for r in rows]
        boots[name] = boot_macro(y, sx, rng)
        graded[name] = {
            "graded": True, "n_experts": len(common),
            "macro_auc": float(np.mean([r["auc"] for r in rows])),
            "macro_auc_ci95": [float(q) for q in np.quantile(boots[name], [0.025, 0.975])],
            "gate_pass": [r["label"] for r in rows if r["gate_pass"]],
        }

    paired = {}
    names = [n for n in boots]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ya = sources[a].set_index("StudyInstanceUID").loc[common]
            yb = sources[b].set_index("StudyInstanceUID").loc[common]
            yt, sa, sb = y[LABEL_COLS].to_numpy(int), ya[LABEL_COLS].to_numpy(float), yb[LABEL_COLS].to_numpy(float)
            d = []
            while len(d) < N_BOOT:
                idx = rng.integers(0, len(yt), len(yt))
                ok = [j for j in range(len(LABEL_COLS)) if 0 < yt[idx, j].sum() < len(idx)]
                if len(ok) < len(LABEL_COLS):
                    continue
                d.append(np.mean([roc_auc_score(yt[idx, j], sa[idx, j]) - roc_auc_score(yt[idx, j], sb[idx, j]) for j in ok]))
            paired[f"{a}-{b}"] = {"delta": graded[a]["macro_auc"] - graded[b]["macro_auc"],
                                  "ci95": [float(q) for q in np.quantile(d, [0.025, 0.975])]}

    nonexp = train.loc[~train["StudyInstanceUID"].isin(experts["StudyInstanceUID"]), "StudyInstanceUID"]
    agree = []
    for i, a in enumerate(sources):
        for b in list(sources)[i + 1:]:
            m = sources[a].merge(sources[b], on="StudyInstanceUID", suffixes=("_a", "_b"))
            m = m[m["StudyInstanceUID"].isin(nonexp)]
            for c in LABEL_COLS:
                agree.append({"a": a, "b": b, "label": c, "n": len(m),
                              "spearman": spearmanr(m[f"{c}_a"], m[f"{c}_b"]).statistic,
                              "pos_rate_a": float((m[f"{c}_a"] > 0.5).mean()),
                              "pos_rate_b": float((m[f"{c}_b"] > 0.5).mean())})

    OUT.mkdir(parents=True, exist_ok=True)
    pl = pd.DataFrame(per_label)
    pl.to_csv(OUT / "public_labels_vs_expert.csv", index=False)
    ag = pd.DataFrame(agree)
    ag.to_csv(OUT / "public_labels_agreement_nonexpert.csv", index=False)
    summary = {"sources": graded, "paired_macro_auc": paired,
               "mean_spearman_nonexpert": ag.groupby(["a", "b"])["spearman"].mean().round(3)
               .reset_index().to_dict("records")}
    (OUT / "public_labels_audit.json").write_text(json.dumps(summary, indent=2))

    print(pl.pivot(index="label", columns="source", values="auc").round(3).to_string())
    print(pl.pivot(index="label", columns="source", values="prec").round(2).to_string())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
