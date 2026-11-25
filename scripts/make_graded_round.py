#!/usr/bin/env python3
"""Label round with severity-graded report labels and a decorrelated teacher (6 Oct).

    python scripts/make_graded_round.py --out labels_r5.parquet [--a 0.3] [--decor D1] \
        --family 'outputs/runpod_results/r2-c224hvy40-f*/*' --family 'outputs/runpod_results/r2-effv2s336-f*/*' ...

target = a * M[grade] + (1 - a) * Platt(teacher*), per finding, for every non-expert study
  grade    = scripts/graded_labels.py (0 absent, 1 not mentioned, 2 minimal / low-grade, 3 definite, 4 severe)
  M        = P(expert positive | grade) pooled over the twelve findings on the 58 experts (five numbers)
  teacher* = clean OOF rank mean of the families (as make_round_labels.py), then the finding-specificity step
             (expert_ruler.DECOR[--decor]: target minus alpha x partner on normal scores), back to a 0..1 rank
  Platt    = logistic fit of teacher* on (grade >= 3), so the teacher half sits on the scale of definite findings
Measured on the 57 experts with report labels (6 Oct, teacher as a 0..1 rank): labels_r4 rule 0.9410, a = 0.25 with
D1 0.9505, a = 0.35 0.9493, a = 0.5 0.9449. Prints the expert AUCs of every part for the built target.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import expert_ruler as er                   # noqa: E402
from make_round_labels import family_rank   # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LAB = er.LAB


def to_z(R: pd.DataFrame) -> pd.DataFrame:
    return R.apply(lambda c: norm.ppf((rankdata(c) - 0.5) / len(c)))


def to_rank(Z: pd.DataFrame) -> pd.DataFrame:
    return Z.apply(lambda c: rankdata(c) / len(c))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", action="append", required=True)
    ap.add_argument("--graded", default="labels_graded_v1.parquet")
    ap.add_argument("--base", default="labels_r4.parquet", help="file whose rows / expert rows are kept as the frame")
    ap.add_argument("--out", required=True)
    ap.add_argument("--a", type=float, default=0.3, help="weight of the graded report label")
    ap.add_argument("--grade-map", default="0.03,0.10,0.34,0.79,0.90")
    ap.add_argument("--decor", default="D1")
    ap.add_argument("--min-cover", type=float, default=0.98, help="required share of non-expert studies with OOF + grade")
    ap.add_argument("--mix", default="logit", choices=["logit", "prob"],
                    help="logit: target = sigmoid(a * logit(M[grade]) + (1 - a) * Platt logit) - the teacher enters linearly "
                         "in its rank, as in the expert measurement; prob: a * M[grade] + (1 - a) * Platt probability")
    a = ap.parse_args()
    M = np.array([float(x) for x in a.grade_map.split(",")])
    assert len(M) == 5 and (np.diff(M) > 0).all()
    spec = er.DECOR.get(a.decor, [])

    base = pd.read_parquet(REPO / "data/processed" / a.base)
    base["StudyInstanceUID"] = base["StudyInstanceUID"].astype(str)
    base = base.set_index("StudyInstanceUID")[LAB].astype(float)
    G = pd.read_parquet(REPO / "data/processed" / a.graded)
    G["StudyInstanceUID"] = G["StudyInstanceUID"].astype(str)
    G = G.set_index("StudyInstanceUID")[LAB].astype(int)
    tr = pd.read_csv(REPO / "data/raw/train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    gold = tr[LAB].dropna().astype(int)

    oof = [family_rank(p, "raptor_oof_*_fold[0-9].npz") for p in a.family]
    ids = sorted(set.intersection(*(set(o.index) for o in oof)) & set(base.index) & set(G.index) - set(gold.index))
    nonexp = [u for u in base.index if u not in gold.index]
    print(f"families {len(oof)} | studies with OOF + grade {len(ids)} of {len(nonexp)} non-expert")
    if len(ids) < a.min_cover * len(nonexp):
        raise SystemExit(f"coverage below {a.min_cover}: grades or folds missing - labels NOT written")
    E = sum(o.loc[ids] for o in oof) / len(oof)
    Ed = to_rank(er.decor(to_z(E), spec)) if spec else to_rank(to_z(E))
    Gm = G.loc[ids].apply(lambda c: M[c.values])

    lg = lambda q: np.log(q / (1 - q))

    def mix(gm, lr, e):
        if a.mix == "prob":
            return a.a * gm + (1 - a.a) * lr.predict_proba(e)[:, 1]
        return 1 / (1 + np.exp(-(a.a * lg(gm) + (1 - a.a) * lr.decision_function(e))))

    out = base.copy()
    platt = {}
    print(f"\n{'finding':17s} grades 0/1/2/3/4 (%)        mean target   mean labels_r4   spearman with labels_r4")
    for f in LAB:
        y = (G.loc[ids, f] >= 3).astype(int)
        lr = LogisticRegression(C=10.0).fit(Ed[[f]].values, y)
        platt[f] = lr
        t = mix(Gm[f].values, lr, Ed[[f]].values)
        out.loc[ids, f] = t
        gc = np.bincount(G.loc[ids, f], minlength=5) / len(ids) * 100
        print(f"{f:17s} {' '.join(f'{v:4.0f}' for v in gc)}        {t.mean():.3f}         {base.loc[ids, f].mean():.3f}"
              f"            {pd.Series(t).corr(pd.Series(base.loc[ids, f].values), method='spearman'):.3f}")
    c_new, c_old = out.loc[ids].corr(method="spearman"), base.loc[ids].corr(method="spearman")
    for x, y_ in (("Medial OA", "Lateral OA"), ("Medial OA", "PF OA"), ("Effusion", "Synovitis"), ("Medial Meniscus", "Lateral Meniscus"), ("Contusion", "Fracture")):
        print(f"  column Spearman {x} ~ {y_}: {c_old.loc[x, y_]:.2f} ({a.base}) -> {c_new.loc[x, y_]:.2f}")

    # ---- the same rule on the 58 experts: every model's gold-dev + holdout predictions ranked together (expert_ruler)
    import glob
    import re
    models = []
    for pat in a.family:
        for d in sorted(d for q in pat.split(",") for d in glob.glob(q)):
            gs = [f for f in glob.glob(f"{d}/raptor_gold_*.npz") if not re.search(r"_(swa|ema|last|best|epens|ep\d+)\.npz$", f)]
            hs = [f for f in glob.glob(f"{d}/raptor_holdout_*.npz")]
            if len(gs) == 1 and len(hs) == 1:
                zz = [np.load(f, allow_pickle=True) for f in (gs[0], hs[0])]
                models.append(pd.concat([pd.DataFrame(z["pred"], index=z["ids"].astype(str), columns=LAB) for z in zz]))
    if models:
        ex = [u for u in models[0].index if u in G.index and u in gold.index]
        Ze = er.zscores(models, ex)
        Ee, Eed = to_rank(Ze), (to_rank(er.decor(Ze, spec)) if spec else to_rank(Ze))
        Ge = G.loc[ex].apply(lambda c: M[c.values])
        T = pd.DataFrame({f: mix(Ge[f].values, platt[f], Eed[[f]].values) for f in LAB}, index=ex)
        Y = gold.loc[ex]
        row = lambda S: {f: roc_auc_score(Y[f], S[f]) for f in LAB}
        tab = pd.DataFrame({"grade": row(Ge), "teacher": row(Ee), f"teacher + {a.decor}": row(Eed), "target": row(T)})
        print(f"\n{len(ex)} experts ({len(models)} models), AUC vs expert labels:")
        print(tab.round(3).to_string())
        print("macro:", {c: round(float(tab[c].mean()), 4) for c in tab.columns})
    dest = REPO / "data/processed" / a.out
    out.reset_index().to_parquet(dest, index=False)
    json.dump({"a": a.a, "grade_map": M.tolist(), "decor": a.decor, "spec": spec, "families": a.family, "graded": a.graded, "n": len(ids)},
              open(str(dest).replace(".parquet", ".json"), "w"), indent=1)
    print(f"\nwrote {dest} ({len(ids)} studies re-targeted)")


if __name__ == "__main__":
    main()
