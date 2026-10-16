#!/usr/bin/env python3
"""labels_e1: the teacher's ordering with the EXPERT THRESHOLD of the report severity grade folded in (16 Oct).

    python scripts/make_expert_threshold_labels.py [--a 0.3] [--out labels_e1.parquet]

Why: on the 58 expert studies the experts confirm most moderate / severe report mentions and reject most low-grade
ones (grade 2 vs >= 3: Lateral OA 2/7 vs 7/11, PF OA 6/14 vs 11/11, Lateral Meniscus 3/12 vs 17/19), while every
target we trained on so far (report presence, the expert-trained checkpoints' ordering = labels_t2) ranks low-grade
mentions high. Order per finding = a x M[grade] + (1 - a) x teacher rank, M = P(expert positive | grade) pooled on the
58 (0.03 / 0.10 / 0.34 / 0.79 / 0.90; scripts/graded_labels.py, 6 Oct), teacher = the dump behind labels_t2. Values =
labels_r1's, reassigned in that order (quantile mapping, as labels_t2): only the ORDER changes. Applied only to the
findings where the order agreed better with the experts (58, a = 0.3); the others keep labels_t2's order.
On the 58 (a ruler that has ranked models wrongly before - a check of targets, not a score): macro 0.921 -> 0.932;
Lateral Meniscus 0.878 -> 0.913, Lateral OA 0.793 -> 0.814, PF OA 0.865 -> 0.898, ACL 0.971 -> 0.996, MCL 0.971 -> 0.995.
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
GRADED = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Lateral OA", "PF OA", "Baker's"]
M = np.array([0.03, 0.10, 0.34, 0.79, 0.90])
ap = argparse.ArgumentParser(); ap.add_argument("--a", type=float, default=0.3); ap.add_argument("--out", default="labels_e1.parquet")
a = ap.parse_args()
z = np.load(REPO / "outputs/kernels/teacher-dump-all/_out/teacher_dump.npz", allow_pickle=True)
ids = np.array([str(x) for x in z["ids"]]); T = 0.6 * z["t384m"] + 0.4 * z["t224B"]      # labels_t2's teacher (w224 0.4)
t2 = pd.read_parquet(REPO / "data/processed/labels_t2.parquet"); t2["StudyInstanceUID"] = t2.StudyInstanceUID.astype(str)
r1 = pd.read_parquet(REPO / "data/processed/labels_r1.parquet"); r1["StudyInstanceUID"] = r1.StudyInstanceUID.astype(str)
gr = pd.read_parquet(REPO / "data/processed/labels_graded_v1.parquet"); gr["StudyInstanceUID"] = gr.StudyInstanceUID.astype(str)
u = t2.StudyInstanceUID.values
row = {s: i for i, s in enumerate(ids)}
assert all(s in row for s in u)
Tu = T[[row[s] for s in u]]
G = gr.set_index("StudyInstanceUID").reindex(u)[LAB]
r1v = r1.set_index("StudyInstanceUID").loc[u]
out = t2.copy()
for f in GRADED:
    k = LAB.index(f)
    g = G[f].fillna(1).clip(0, 4).astype(int).values
    order_key = a.a * M[g] + (1 - a.a) * rankdata(Tu[:, k]) / len(u)
    order = np.argsort(np.argsort(order_key, kind="stable"), kind="stable")
    out[f] = np.sort(r1v[f].values)[order].astype(np.float32)
    assert abs(out[f].mean() - r1v[f].mean()) < 1e-5
    print(f"{f:17s} grade known {np.isfinite(G[f].values).mean():.2f} | Spearman to labels_t2 {pd.Series(out[f].values).corr(pd.Series(t2[f].values), method='spearman'):.3f}")
out.to_parquet(REPO / "data/processed" / a.out, index=False)
print(f"wrote data/processed/{a.out}: {len(out)} studies (labels_t2's studies; experts left out)")
