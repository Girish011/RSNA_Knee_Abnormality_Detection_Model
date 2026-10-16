#!/usr/bin/env python3
"""Training targets from the two expert-trained public checkpoints (scores of `teacher-dump-all`).

    python scripts/make_teacher_labels.py [--name teacher-dump-all] [--out labels_t2.parquet] [--w224 0.4]

Order of the studies = (1 - w224) x t384m + w224 x t224B (t384m = plain + mirror pass of the 384 arm); the values are
labels_r1's own, reassigned in that order (quantile mapping per finding). The 58 expert
studies are left out (the trainer keeps them for validation with their expert labels). Written to data/processed/ and
copied into outputs/mega_queue/ (then `python scripts/mega.py push`). Prints how the targets compare with labels_r1.
"""
import argparse, shutil
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[1]
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="teacher-dump-all"); ap.add_argument("--out", default="labels_t2.parquet"); ap.add_argument("--w224", type=float, default=0.4)
a = ap.parse_args()
z = np.load(REPO / "outputs/kernels" / a.name / "_out/teacher_dump.npz", allow_pickle=True)
ids = np.array([str(x) for x in z["ids"]]); assert list(z["lab"]) == LAB
T = (1 - a.w224) * z["t384m"] + a.w224 * z["t224B"]
ok = ~np.isnan(T).any(1)
tr = pd.read_csv(REPO / "data/raw/train.csv"); tr["StudyInstanceUID"] = tr.StudyInstanceUID.astype(str)
gold = set(tr[tr[LAB].notna().all(axis=1)].StudyInstanceUID)
keep = ok & ~np.isin(ids, list(gold))
df = pd.DataFrame(T[keep].astype(np.float32), columns=LAB); df.insert(0, "StudyInstanceUID", ids[keep])
r1 = pd.read_parquet(REPO / "data/processed/labels_r1.parquet"); r1["StudyInstanceUID"] = r1.StudyInstanceUID.astype(str)
df = df[df.StudyInstanceUID.isin(set(r1.StudyInstanceUID))].reset_index(drop=True)      # same studies as labels_r1
# Same marginal distribution as labels_r1, finding by finding: the i-th lowest teacher score gets the i-th lowest
# labels_r1 value. The paired run then differs from its control (labels_r1) only in the ORDER of the studies, not in
# prevalence, loss scale or positive weights (raw teacher scores average 0.39-0.50 on every finding).
raw = df.copy()
r1s = r1.set_index("StudyInstanceUID").loc[df.StudyInstanceUID]
for l in LAB:
    order = np.argsort(np.argsort(raw[l].values, kind="stable"), kind="stable")
    df[l] = np.sort(r1s[l].values)[order].astype(np.float32)
assert all(abs(df[l].mean() - r1s[l].mean()) < 1e-5 for l in LAB)
m = df.merge(r1, on="StudyInstanceUID", suffixes=("", "_r1"))
print(f"scored {int(ok.sum())} of {len(ids)} | targets for {len(df)} studies (labels_r1 has {len(r1)})")
print(f"{'finding':18s} mean_t2 mean_r1  spearman(t2, r1)")
for l in LAB:
    print(f"{l:18s} {m[l].mean():7.3f} {m[l + '_r1'].mean():7.3f}  {m[l].corr(m[l + '_r1'], method='spearman'):.3f}")
assert df[LAB].notna().all().all() and df[LAB].values.min() >= 0 and df[LAB].values.max() <= 1
df.to_parquet(REPO / "data/processed" / a.out, index=False)
shutil.copy2(REPO / "data/processed" / a.out, REPO / "outputs/mega_queue" / a.out)
print(f"wrote data/processed/{a.out} and outputs/mega_queue/{a.out}")
