#!/usr/bin/env python3
"""Do DICOM header tags (PatientSex, PatientAge, scanner) add to the image model?

    python scripts/meta_prior_test.py [--meta outputs/meta_dump/meta_train.csv] [--family r1-c224:r1c224]

Image evidence = 5-fold OOF of one family (4,349 non-expert studies, each predicted by a model that never saw it).
Per finding, a cross-fitted logistic stacker on [logit(p_img), header features] is compared with p_img alone:
  said   = macro AUC vs the report where it explicitly says YES / NO (pilk verdicts), bootstrap SE of the difference
  meta   = AUC of the header features alone (no pixels)
The header features are label-neutral, so `said` is a valid ruler for them (docs/STATUS 2026-10-04 17:55).
Expert check (58 studies, noise 0.01-0.02): prevalence by sex next to the report prevalence, sign check only.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

REPO = Path(__file__).resolve().parents[1]
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def load_oof(job: str, tag: str) -> pd.DataFrame:
    parts = []
    for k in range(5):
        fs = glob.glob(str(REPO / f"outputs/runpod_results/{job}-f{k}*/{tag}{k}/raptor_oof_{tag}{k}_fold{k}.npz"))
        z = np.load(fs[0], allow_pickle=True)
        p = pd.DataFrame(z["pred"], index=z["ids"].astype(str), columns=LAB)
        parts.append(p.rank(pct=True))          # rank inside the fold: folds share one scale
    return pd.concat(parts)


def features(meta: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=meta.index)
    sex = meta.get("PatientSex", pd.Series("", index=meta.index)).fillna("").str.upper().str[:1]
    X["sex_m"] = (sex == "M").astype(float); X["sex_f"] = (sex == "F").astype(float)
    age = pd.to_numeric(meta.get("PatientAge", pd.Series("", index=meta.index)).fillna("").astype(str)
                        .str.extract(r"(\d+)")[0], errors="coerce")
    X["age_missing"] = age.isna().astype(float)
    a = age.fillna(age.median() if age.notna().any() else 0.0)
    X["age"] = (a - 45.0) / 20.0; X["age2"] = X["age"] ** 2
    for col in ("PatientWeight", "PatientSize"):
        v = pd.to_numeric(meta.get(col, pd.Series(np.nan, index=meta.index)), errors="coerce")
        X[col + "_missing"] = v.isna().astype(float)
        X[col] = ((v - v.median()) / (v.std() + 1e-6)).fillna(0.0) if v.notna().any() else 0.0
    man = meta.get("Manufacturer", pd.Series("", index=meta.index)).fillna("").str.upper().str[:4]
    fs = pd.to_numeric(meta.get("MagneticFieldStrength", pd.Series(np.nan, index=meta.index)), errors="coerce")
    fs = fs.where(fs < 100, fs / 10000.0).round(1)      # some vendors write gauss
    grp = man + "_" + fs.astype(str)
    for g, n in grp.value_counts().items():
        if n >= 40:
            X["scan_" + g] = (grp == g).astype(float)
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="outputs/meta_dump/meta_train.csv")
    ap.add_argument("--family", default="r1-c224:r1c224")
    ap.add_argument("--C", type=float, default=1.0)
    a = ap.parse_args()
    meta = pd.read_csv(REPO / a.meta, dtype=str).set_index("StudyInstanceUID")
    job, tag = a.family.split(":")
    P = load_oof(job, tag)
    pk = pd.read_csv(REPO / "data/external/public_labels/pilk/report_labels_v2.csv",
                     dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    X = features(meta).reindex(P.index).fillna(0.0)
    groups = {"sex": [c for c in X if c.startswith("sex_")],
              "age": [c for c in X if c.startswith("age")],
              "body": [c for c in X if c.startswith("Patient")],
              "scanner": [c for c in X if c.startswith("scan_")]}
    print(f"studies {len(P)} | header columns {list(X.columns)}")
    print({k: int((X[v].abs().sum(1) > 0).sum()) for k, v in groups.items() if v})

    def cv_pred(F: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
        out = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(F):
            m = LogisticRegression(C=a.C, max_iter=2000).fit(F[tr], y[tr])
            out[te] = m.decision_function(F[te])
        return out

    rng = np.random.default_rng(0)
    rows, boots = [], {}
    sets = {"all": list(X.columns), **{k: v for k, v in groups.items() if v}}
    for f in LAB:
        v = pk[f + "__verdict"].reindex(P.index)
        m = v.isin(["YES", "NO"]).values
        y = (v[m] == "YES").astype(int).values
        p = P[f].values[m].clip(1e-4, 1 - 1e-4)
        lg = np.log(p / (1 - p))[:, None]
        r = {"finding": f, "n": int(m.sum()), "prev": round(float(y.mean()), 3), "img": roc_auc_score(y, p)}
        for name, cols in sets.items():
            Xm = X.loc[m, cols].values
            s = cv_pred(np.hstack([lg, Xm]), y)
            r["img+" + name] = roc_auc_score(y, s)
            if name == "all":
                r["meta_only"] = roc_auc_score(y, cv_pred(Xm, y))
                boots[f] = (y, p, s)
        rows.append(r)
    R = pd.DataFrame(rows).set_index("finding")
    pd.set_option("display.width", 220)
    print(R.round(4).to_string())
    mac = R.drop(columns=["n", "prev"]).mean()
    print("\nmacro:", mac.round(4).to_dict())
    d = []
    for _ in range(300):
        dd = []
        for f, (y, p, s) in boots.items():
            i = rng.integers(0, len(y), len(y))
            if 0 < y[i].sum() < len(i):
                dd.append(roc_auc_score(y[i], s[i]) - roc_auc_score(y[i], p[i]))
        d.append(np.mean(dd))
    print(f"img+all - img on `said`: {mac['img+all'] - mac['img']:+.4f} +- {np.std(d):.4f}")

    tr = pd.read_csv(REPO / "data/raw/train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    gold = tr[tr["ACL"].notna()]
    Xg = features(meta).reindex(gold.index)
    sex = np.where(Xg["sex_m"] > 0, "M", np.where(Xg["sex_f"] > 0, "F", "?"))
    print("\nexperts by sex (prevalence, n):", pd.Series(sex).value_counts().to_dict())
    sx = features(meta).reindex(P.index)
    out = []
    for f in LAB:
        v = pk[f + "__verdict"].reindex(P.index)
        ok = v.isin(["YES", "NO"]).values
        yes = (v == "YES").values
        out.append({"finding": f,
                    "expert M": gold[f][sex == "M"].mean(), "expert F": gold[f][sex == "F"].mean(),
                    "report M": yes[ok & (sx["sex_m"].values > 0)].mean(),
                    "report F": yes[ok & (sx["sex_f"].values > 0)].mean()})
    print(pd.DataFrame(out).set_index("finding").round(3).to_string())


if __name__ == "__main__":
    main()
