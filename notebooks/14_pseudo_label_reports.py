# Kaggle GPU T4x2 + Internet ON — train-only report pseudo-labels (weak_labels_v3)
# Public model: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli  (not used at test time)
#
# Attach: competition + rsna-knee-code (for src/ + weak_labels_v1.csv)
# Settings: GPU T4 x2, Internet ON. Do not pip install torch.
# Save Version: weak-labels-v3-zeroshot
# Then download /kaggle/working/weak_labels_v3.csv
#
# v3 upgrade: ensemble several paraphrased entailment hypotheses per label (steadier
# on multilingual FR/ES/DE reports) and reuse the unit-tested merge helper
# (NLI decision > v1 keyword fallback > expert override) from rsna_knee.text.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import pipeline

print("cuda", torch.cuda.is_available(), torch.__version__, flush=True)

# Import the tested package from the attached rsna-knee-code dataset.
CODE = Path("/kaggle/input/datasets/girishbose/rsna-knee-code")
for cand in [CODE / "src", *[(p.parent) for p in Path("/kaggle/input").rglob("rsna_knee/__init__.py")]]:
    if (Path(cand) / "rsna_knee/__init__.py").exists() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))
        break

from rsna_knee.constants import LABEL_COLS  # noqa: E402
from rsna_knee.text.zeroshot_labels import (  # noqa: E402
    HYPOTHESIS_VARIANTS,
    aggregate_variant_scores,
    impression_text,
    merge_pseudo_labels,
    scores_to_label,
)

COMP = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
if not (COMP / "train.csv").exists():
    hits = list(Path("/kaggle/input").rglob("train.csv"))
    COMP = next(h.parent for h in hits if "rsna-knee" in str(h).lower())

train = pd.read_csv(COMP / "train.csv")
print("studies", len(train), flush=True)

# Flatten all hypothesis variants; remember which label each belongs to.
candidate_labels = []
variant_to_label = {}
for label, variants in HYPOTHESIS_VARIANTS.items():
    for v in variants:
        candidate_labels.append(v)
        variant_to_label[v] = label

device = 0 if torch.cuda.is_available() else -1
clf = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    device=device,
)

rows = []
texts = [impression_text(r) for r in train["Report"].astype(str).tolist()]
uids = train["StudyInstanceUID"].astype(str).tolist()
BATCH = 4
for i in range(0, len(texts), BATCH):
    chunk = texts[i : i + BATCH]
    outs = clf(
        chunk,
        candidate_labels=candidate_labels,
        hypothesis_template="This radiology report states that {}.",
        multi_label=True,
    )
    if isinstance(outs, dict):
        outs = [outs]
    for uid, out in zip(uids[i : i + BATCH], outs):
        rec = {"StudyInstanceUID": uid}
        score_map = dict(zip(out["labels"], out["scores"]))
        # Aggregate variant scores per label, then threshold.
        per_label_scores: dict[str, list[float]] = {c: [] for c in LABEL_COLS}
        for hyp, lab in variant_to_label.items():
            per_label_scores[lab].append(float(score_map.get(hyp, 0.0)))
        for lab in LABEL_COLS:
            agg = aggregate_variant_scores(per_label_scores[lab], method="mean")
            val, conf = scores_to_label(agg)
            rec[lab] = val
            rec[f"{lab}__conf"] = conf
            rec[f"{lab}__zs"] = agg
        rows.append(rec)
    if i % 100 == 0:
        print(f"labeled {min(i + BATCH, len(texts))}/{len(texts)}", flush=True)

zs = pd.DataFrame(rows)

# Expert audit: precision on labels the NLI model committed to (non-abstain).
expert = train.loc[train[LABEL_COLS].notna().all(axis=1), ["StudyInstanceUID"] + LABEL_COLS]
m = expert.merge(zs, on="StudyInstanceUID", suffixes=("_gold", "_pred"))
audit_rows = []
for c in LABEL_COLS:
    y = m[f"{c}_gold"].astype(int).to_numpy()
    pred = m[f"{c}_pred"]
    known = pred.notna().to_numpy()
    bin_pred = (pred == 1).fillna(False).astype(int).to_numpy()
    audit_rows.append(
        {
            "label": c,
            "n_known": int(known.sum()),
            "prec_known": float(precision_score(y[known], bin_pred[known], zero_division=0)) if known.any() else np.nan,
            "rec_all": float(recall_score(y, bin_pred, zero_division=0)),
            "f1_known": float(f1_score(y[known], bin_pred[known], zero_division=0)) if known.any() else np.nan,
        }
    )
audit = pd.DataFrame(audit_rows)
print(audit.to_string(index=False), flush=True)
print("macro f1_known", float(audit["f1_known"].mean()), "macro prec_known", float(audit["prec_known"].mean()), flush=True)

# Layer: NLI decision > v1 keyword fallback (conf>=0.5) > expert override.
v1_path = CODE / "data/processed/weak_labels_v1.csv"
v1 = pd.read_csv(v1_path) if v1_path.exists() else None
expert_full = train.loc[train[LABEL_COLS].notna().all(axis=1), ["StudyInstanceUID"] + LABEL_COLS]
out = merge_pseudo_labels(zs, v1=v1, expert=expert_full, min_v1_conf=0.5)

keep = ["StudyInstanceUID"] + LABEL_COLS + [f"{c}__conf" for c in LABEL_COLS]
dest = Path("/kaggle/working/weak_labels_v3.csv")
out[keep].to_csv(dest, index=False)
print("wrote", dest, "any-label", int(out[LABEL_COLS].notna().any(axis=1).sum()), "/", len(out), flush=True)
audit.to_csv("/kaggle/working/weak_label_v3_vs_expert.csv", index=False)
print("done", flush=True)
