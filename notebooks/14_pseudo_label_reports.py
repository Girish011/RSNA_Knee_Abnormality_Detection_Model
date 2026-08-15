# Kaggle GPU T4x2 + Internet ON — train-only report pseudo-labels (weak_labels_v3)
# Public model: MoritzLaurer/mDeBERTa-v3-base-mnli-xnli  (not used at test time)
#
# Attach: competition + rsna-knee-code (for weak_labels_v1.csv)
# Settings: GPU T4 x2, Internet ON. Do not pip install torch.
# Save Version: weak-labels-v3-zeroshot
# Then download /kaggle/working/weak_labels_v3.csv

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import pipeline

print("cuda", torch.cuda.is_available(), torch.__version__, flush=True)

LABEL_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]
HYPOTHESES = {
    "ACL": "an ACL or anterior cruciate ligament tear or rupture is present",
    "MCL": "an MCL or medial collateral ligament tear or sprain is present",
    "Medial Meniscus": "a medial meniscus tear is present",
    "Lateral Meniscus": "a lateral meniscus tear is present",
    "Medial OA": "medial compartment osteoarthritis or cartilage loss is present",
    "Lateral OA": "lateral compartment osteoarthritis or cartilage loss is present",
    "PF OA": "patellofemoral osteoarthritis or cartilage loss is present",
    "Effusion": "a knee joint effusion is present",
    "Synovitis": "synovitis is present",
    "Baker's": "a Baker cyst or popliteal cyst is present",
    "Contusion": "a bone contusion or bone bruise is present",
    "Fracture": "a fracture is present",
}
POS_TH, NEG_TH = 0.70, 0.30

COMP = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
CODE = Path("/kaggle/input/datasets/girishbose/rsna-knee-code")
if not (COMP / "train.csv").exists():
    hits = list(Path("/kaggle/input").rglob("train.csv"))
    COMP = next(h.parent for h in hits if "rsna-knee" in str(h).lower())

train = pd.read_csv(COMP / "train.csv")
print("studies", len(train), flush=True)


def impression_text(report: str, max_chars: int = 1800) -> str:
    t = " ".join(str(report or "").split())
    return t if len(t) <= max_chars else t[-max_chars:]


def scores_to_label(score: float):
    s = float(score)
    if s >= POS_TH:
        return 1.0, s
    if s <= NEG_TH:
        return 0.0, 1.0 - s
    return None, s


device = 0 if torch.cuda.is_available() else -1
clf = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    device=device,
)
candidate_labels = list(HYPOTHESES.values())
label_from_hyp = {v: k for k, v in HYPOTHESES.items()}

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
        for hyp, lab in label_from_hyp.items():
            val, conf = scores_to_label(float(score_map.get(hyp, 0.0)))
            rec[lab] = val
            rec[f"{lab}__conf"] = conf
            rec[f"{lab}__zs"] = float(score_map.get(hyp, 0.0))
        rows.append(rec)
    if i % 100 == 0:
        print(f"labeled {min(i + BATCH, len(texts))}/{len(texts)}", flush=True)

zs = pd.DataFrame(rows)

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

out = zs.copy()
v1_path = CODE / "data/processed/weak_labels_v1.csv"
if v1_path.exists():
    v1 = pd.read_csv(v1_path).set_index("StudyInstanceUID")
    for i, row in out.iterrows():
        uid = str(row["StudyInstanceUID"])
        if uid not in v1.index:
            continue
        vr = v1.loc[uid]
        for c in LABEL_COLS:
            if pd.isna(out.at[i, c]) and c in vr.index and pd.notna(vr[c]):
                conf = float(vr[f"{c}__conf"]) if f"{c}__conf" in vr.index and pd.notna(vr[f"{c}__conf"]) else 0.0
                if conf >= 0.5:
                    out.at[i, c] = vr[c]
                    out.at[i, f"{c}__conf"] = conf

em = train.set_index("StudyInstanceUID")
for i, row in out.iterrows():
    uid = str(row["StudyInstanceUID"])
    if uid not in em.index:
        continue
    er = em.loc[uid]
    if not er[LABEL_COLS].notna().all():
        continue
    for c in LABEL_COLS:
        out.at[i, c] = er[c]
        out.at[i, f"{c}__conf"] = 1.0

keep = ["StudyInstanceUID"] + LABEL_COLS + [f"{c}__conf" for c in LABEL_COLS]
dest = Path("/kaggle/working/weak_labels_v3.csv")
out[keep].to_csv(dest, index=False)
print("wrote", dest, "any-label", int(out[LABEL_COLS].notna().any(axis=1).sum()), "/", len(out), flush=True)
audit.to_csv("/kaggle/working/weak_label_v3_vs_expert.csv", index=False)
print("done", flush=True)
