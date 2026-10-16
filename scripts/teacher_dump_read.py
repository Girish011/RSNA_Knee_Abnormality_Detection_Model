#!/usr/bin/env python3
"""Read outputs/kernels/<name>/_out/teacher_dump.npz: which 224 input is right, are the scores memorised report labels,
and where do the expert-trained models differ from ours. Everything here is offline (not leaderboard).

    python scripts/teacher_dump_read.py [--name teacher-dump]
"""
import argparse, glob, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PAIRS = [("Effusion", "Synovitis"), ("Contusion", "Fracture"), ("Medial OA", "Lateral OA"), ("Medial OA", "PF OA"),
         ("Lateral OA", "PF OA"), ("Medial Meniscus", "Lateral Meniscus")]
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="teacher-dump"); a = ap.parse_args()
z = np.load(REPO / "outputs/kernels" / a.name / "_out/teacher_dump.npz", allow_pickle=True)
ids = [str(x) for x in z["ids"]]; arms = [k for k in z.files if k.startswith("t")]
tr = pd.read_csv(REPO / "data/raw/train.csv"); tr["StudyInstanceUID"] = tr.StudyInstanceUID.astype(str)
gold = tr[tr[LAB].notna().all(axis=1)].set_index("StudyInstanceUID")
rep = pd.read_csv(REPO / "data/external_labels/stevenleehans_rsna-knee-llm-report-labels/llm_labels_v4_blend.csv")
rep["StudyInstanceUID"] = rep.StudyInstanceUID.astype(str); rep = rep.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
# our own models on the same studies: fifteen heavy-aug models, out-of-fold for the report-labelled studies, gold-val + holdout for the experts
own = {}
for fam in ("r2-c224hvy40", "r2-effv2s336", "r2-cnxs336"):
    for d in sorted(glob.glob(str(REPO / "outputs/runpod_results") + f"/{fam}-f*/*/")):
        for f in glob.glob(d + "raptor_oof_*_fold?.npz") + glob.glob(d + "raptor_gold_*.npz") + glob.glob(d + "raptor_holdout_*.npz"):
            if any(k in f for k in ("_best_", "_ema", "_ep", "_last_", "_swa", "epens")): continue
            y = np.load(f, allow_pickle=True)
            for u, p in zip(y["ids"], y["pred"]): own.setdefault(str(u), []).append(p)
ok = ~np.isnan(z["t384"]).any(1)
gi = [i for i, u in enumerate(ids) if u in gold.index and ok[i]]
ri = [i for i, u in enumerate(ids) if u not in gold.index and u in rep.index and u in own and ok[i]]
print(f"studies {len(ids)} | scored {int(ok.sum())} | experts {len(gi)} | report-labelled with own OOF {len(ri)}")
Yg = gold.loc[[ids[i] for i in gi], LAB].values
Rb = (rep.loc[[ids[i] for i in ri], LAB].values >= 0.5).astype(int)
P = {k: z[k] for k in arms}
P["own15"] = np.stack([np.mean(own[u], 0) if u in own else np.full(12, np.nan) for u in ids])
def rk(x): return np.apply_along_axis(lambda v: pd.Series(v).rank(pct=True).values, 0, x)
if "t224B" in P: P["t384m+224B"] = 0.6 * rk(np.nan_to_num(P["t384m"], nan=0.5)) + 0.4 * rk(np.nan_to_num(P["t224B"], nan=0.5))
print("\nA. Agreement with t384m (mean Spearman over findings, all scored studies) and the 58 experts (per finding)")
print(f"{'arm':12s} rho    macro | " + " ".join(f"{l[:7]:>7s}" for l in LAB))
for k, v in P.items():
    rho = np.mean([spearmanr(v[ok, j], P["t384m"][ok, j], nan_policy="omit").correlation for j in range(12)])
    au = [roc_auc_score(Yg[:, j], v[gi, j]) for j in range(12)]
    print(f"{k:12s} {rho:.3f}  {np.mean(au):.4f} | " + " ".join(f"{x:7.3f}" for x in au))
print(f"\nB. Against the report verdicts on {len(ri)} training studies (teachers saw these in training; own15 is out-of-fold)")
print(f"{'arm':12s} macro | " + " ".join(f"{l[:7]:>7s}" for l in LAB))
for k, v in P.items():
    au = [roc_auc_score(Rb[:, j], v[ri, j]) for j in range(12)]
    print(f"{k:12s} {np.mean(au):.4f} | " + " ".join(f"{x:7.3f}" for x in au))
print("\nC. Telling paired findings apart (studies where the report has exactly one of the two) and column correlation")
print(f"{'arm':12s} " + " ".join(f"{a[:3]}/{b[:3]:<5s}" for a, b in [(x.replace('Medial ', 'M').replace('Lateral ', 'L'), y.replace('Medial ', 'M').replace('Lateral ', 'L')) for x, y in PAIRS]))
for k, v in P.items():
    out = []
    for x, y in PAIRS:
        i, j = LAB.index(x), LAB.index(y); m = Rb[:, i] != Rb[:, j]
        out.append(f"{roc_auc_score(Rb[m, i], rk(v[ri])[m, i] - rk(v[ri])[m, j]):.3f}/{spearmanr(v[ri, i], v[ri, j]).correlation:+.2f}")
    print(f"{k:12s} " + " ".join(out))
print("   experts' own label correlation on the 58: " + " ".join(f"{np.corrcoef(Yg[:, LAB.index(x)], Yg[:, LAB.index(y)])[0, 1]:+.2f}" for x, y in PAIRS))
