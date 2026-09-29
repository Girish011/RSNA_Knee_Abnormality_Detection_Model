#!/usr/bin/env python3
"""Expert-anchored label fusion (self-training): report label + out-of-fold image prediction
→ estimate of the EXPERT label, per finding.

Why: experts label from the images. On the 58 experts the report-derived label scores macro 0.870
while our image members score 0.932 — the training target is worse than the student. We fit, per
finding on the 29 expert-dev studies, a 2-feature L2 logistic model
    P(expert=1) = σ(a·logit(report) + b·logit(image) + c)
check it on the 29 untouched holdout experts, and (with --write) relabel every non-expert train
study from its report label + its OUT-OF-FOLD image prediction.

    python scripts/fuse_labels.py --image-node 5bc0a0f45d75                # fold-0 smoke test
    python scripts/fuse_labels.py --oof-job f-xxxxxxxxxx --write fused_v1  # full 5-fold relabel
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from rsna_knee.constants import LABEL_COLS  # noqa: E402

RESULTS = REPO / "outputs" / "runpod_results"
EPS = 1e-4


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _load(pattern: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(RESULTS / pattern)))
    if not fs:
        raise SystemExit(f"no files for {pattern}")
    frames = []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        frames.append(pd.DataFrame(z["pred"], index=z["ids"].astype(str), columns=LABEL_COLS))
    df = pd.concat(frames)
    return df.groupby(level=0).mean()   # experts are predicted by every fold model → average


def image_preds(node: str | None, oof_job: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(non-expert OOF predictions, expert predictions)."""
    if oof_job:
        oof = _load(f"{oof_job}*/f*/raptor_oof_f*_fold[0-4].npz")
        exp = pd.concat([_load(f"{oof_job}*/f*/raptor_holdout_f*.npz"),
                         _gold(f"{oof_job}*/f*")]).groupby(level=0).mean()
        return oof, exp
    fs = glob.glob(str(RESULTS / f"*/{node}/raptor_oof_{node}_fold0.npz"))
    oof = _load(f"*/{node}/raptor_oof_{node}_fold0.npz") if fs else pd.DataFrame(columns=LABEL_COLS)
    exp = pd.concat([_load(f"*/{node}/raptor_holdout_{node}.npz"), _gold(f"*/{node}")])
    return oof, exp.groupby(level=0).mean()


def _gold(dir_pattern: str) -> pd.DataFrame:
    """raptor_gold_<tag>.npz = dev-expert predictions (ids may be absent in older files)."""
    split = json.loads((REPO / "agent" / "protocol" / "expert_split.json").read_text())
    frames = []
    for f in sorted(glob.glob(str(RESULTS / dir_pattern / "raptor_gold_*.npz"))):
        if "_swa" in f or "_ema" in f or "smoke" in f:
            continue
        z = np.load(f, allow_pickle=True)
        ids = z["ids"].astype(str) if "ids" in z.files else np.array(split["dev"])
        if len(ids) != len(z["pred"]):
            continue
        frames.append(pd.DataFrame(z["pred"], index=ids, columns=LABEL_COLS))
    if not frames:
        raise SystemExit(f"no gold (dev) predictions under {dir_pattern}")
    return pd.concat(frames).groupby(level=0).mean()


def fit_eval(report: pd.DataFrame, image: pd.DataFrame, gold: pd.DataFrame, dev: list[str],
             hold: list[str], C: float = 1.0):
    models, rows = {}, []
    for l in LABEL_COLS:
        def X(ids):
            return np.c_[logit(report.loc[ids, l]), logit(image.loc[ids, l])]
        yd, yh = gold.loc[dev, l].astype(int), gold.loc[hold, l].astype(int)
        m = LogisticRegression(C=C).fit(X(dev), yd)
        fh = m.predict_proba(X(hold))[:, 1]
        auc = lambda s: roc_auc_score(yh, s) if 0 < yh.sum() < len(yh) else np.nan  # noqa: E731
        rows.append({"finding": l, "report": auc(report.loc[hold, l]), "image": auc(image.loc[hold, l]),
                     "fused": auc(fh), "w_report": m.coef_[0][0], "w_image": m.coef_[0][1]})
        models[l] = m
    return models, pd.DataFrame(rows).set_index("finding")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-node", help="online node with fold-0 OOF (smoke test)")
    ap.add_argument("--oof-job", help="members 5-fold OOF job id (f-...)")
    ap.add_argument("--report", default="pilk", help="report-label source available for experts too")
    ap.add_argument("--write", help="write data/processed/labels_<name>.parquet and register it")
    ap.add_argument("--C", type=float, default=1.0)
    a = ap.parse_args()
    train = pd.read_csv(REPO / "data" / "raw" / "train.csv").set_index("StudyInstanceUID")
    gold = train[train[LABEL_COLS].notna().all(axis=1)][LABEL_COLS]
    split = json.loads((REPO / "agent" / "protocol" / "expert_split.json").read_text())
    report = pd.read_csv(REPO / "data" / "external" / "public_labels" / "pilk" / "report_labels_v2.csv"
                         ).set_index("StudyInstanceUID")[LABEL_COLS]
    oof, exp = image_preds(a.image_node, a.oof_job)
    dev = [u for u in split["dev"] if u in exp.index and u in report.index]
    hold = [u for u in split["holdout"] if u in exp.index and u in report.index]
    models, tab = fit_eval(report, exp, gold, dev, hold, a.C)
    pd.set_option("display.width", 140)
    print(f"fit on dev={len(dev)} experts, scored on holdout={len(hold)} (AUC vs expert label)")
    print(tab.round(3).to_string())
    print("macro  report %.3f  image %.3f  fused %.3f" % tuple(tab[c].mean() for c in ("report", "image", "fused")))
    if not a.write:
        return
    ids = [u for u in oof.index if u in report.index]
    fused = pd.DataFrame(index=ids, columns=LABEL_COLS, dtype=float)
    for l, m in models.items():
        fused[l] = m.predict_proba(np.c_[logit(report.loc[ids, l]), logit(oof.loc[ids, l])])[:, 1]
    out = fused.rename_axis("StudyInstanceUID").reset_index()
    fname = f"labels_{a.write}.parquet"
    out.to_parquet(REPO / "data" / "processed" / fname, index=False)
    reg_f = REPO / "agent" / "labels" / "registry.json"
    reg = json.loads(reg_f.read_text())
    reg[a.write] = {"file": fname, "rule": "expert-anchored fusion (report + OOF image)",
                    "source": a.oof_job or a.image_node,
                    "audit": {l: {"holdout_fused": round(float(tab.loc[l, "fused"]), 3),
                                  "holdout_report": round(float(tab.loc[l, "report"]), 3), "kept": True}
                              for l in LABEL_COLS}}
    reg_f.write_text(json.dumps(reg, indent=1))
    print(f"wrote {fname} ({len(out)} studies) and registered {a.write}")


if __name__ == "__main__":
    main()
