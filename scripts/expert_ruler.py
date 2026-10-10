#!/usr/bin/env python3
"""Expert ruler: macro AUC on the 58 expert-graded train studies (never trained on) for runs and their rank means,
with and without the finding-specificity post-processing (`build_ensemble_kernel.py --decor`).

    python scripts/expert_ruler.py --runs r4-cnxs336-f0:r4cnxs3360,r4-cnxsseq-f0:r4cnxsseq0 [--blend] [--decor D1]
    python scripts/expert_ruler.py --group old=r4-cnxs336-f0:r4cnxs3360,... --group new=... [--decor D1]

Each run contributes its gold-dev (29, also used to pick its best epoch) and holdout (29) predictions; missing
`raptor_gold_<tag>.npz` files are fetched from the run's result dataset. Noise: ~+-0.01-0.02 on a macro AUC and
~+-0.003 on a paired difference between two treatments of the SAME predictions (bootstrap printed). It is the only
offline reading against expert labels; `said` (oof_ruler / family_ruler / side_ruler) scores agreement with reports.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
R = REPO / "outputs" / "runpod_results"
KAGGLE = str(Path(sys.executable).with_name("kaggle"))
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
DECOR = {   # name -> [(target, partners, alpha)]; D1 / D2 = the settings submitted on 6 Oct (refs 56878527 / 56878688)
    "D1": [("Lateral OA", ["Medial OA"], 0.3), ("PF OA", ["Medial OA", "Lateral OA"], 0.3), ("Synovitis", ["Effusion"], 0.3),
           ("Lateral Meniscus", ["Medial Meniscus"], 0.2)],
    "D2": [("Lateral OA", ["Medial OA"], 0.5), ("PF OA", ["Medial OA", "Lateral OA"], 0.5), ("Synovitis", ["Effusion"], 0.5),
           ("Lateral Meniscus", ["Medial Meniscus"], 0.3)],
}


def expert_preds(spec: str) -> pd.DataFrame:
    job, tag = spec.split(":")
    d = R / job / tag
    parts = []
    for kind in ("gold", "holdout"):
        f = d / f"raptor_{kind}_{tag}.npz"
        if not f.exists():
            d.mkdir(parents=True, exist_ok=True)
            for name in (f.name, f"{tag}/{f.name}"):
                subprocess.run([KAGGLE, "datasets", "download", f"girishbose/rsna-knee-rp-{job}-out", "-f", name, "-p", str(d), "-q"],
                               capture_output=True, text=True, timeout=300)
                z = d / (f.name + ".zip")
                if z.exists():
                    zipfile.ZipFile(z).extractall(d); z.unlink()
                if f.exists():
                    break
        if not f.exists():
            raise SystemExit(f"{f} missing (not in girishbose/rsna-knee-rp-{job}-out either)")
        z = np.load(f, allow_pickle=True)
        parts.append(pd.DataFrame(z["pred"], index=z["ids"].astype(str), columns=LAB))
    return pd.concat(parts)


def zscores(P: list[pd.DataFrame], idx) -> pd.DataFrame:
    rk = sum(p.loc[idx].rank(pct=True) for p in P) / len(P)
    return rk.apply(lambda c: norm.ppf((rankdata(c) - 0.5) / len(c)))


def decor(Z: pd.DataFrame, spec) -> pd.DataFrame:
    out = Z.copy()
    for t, ps, a in spec:
        out[t] = Z[t] - a * Z[ps].mean(1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="", help="comma list of job:tag, each scored alone")
    ap.add_argument("--group", action="append", default=[], help="name=job:tag,job:tag,... scored as one rank mean")
    ap.add_argument("--blend", action="store_true", help="also the rank mean of all --runs")
    ap.add_argument("--decor", default="D1", help="D1 | D2 | none")
    a = ap.parse_args()
    tr = pd.read_csv(REPO / "data/raw/train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    Y = tr.dropna(subset=LAB)[LAB]
    sets: dict[str, list[pd.DataFrame]] = {}
    for spec in [s for s in a.runs.split(",") if s]:
        sets[spec.split(":")[1]] = [expert_preds(spec)]
    if a.blend and len(sets) > 1:
        sets["blend(runs)"] = [p[0] for p in sets.values()]
    for g in a.group:
        name, specs = g.split("=", 1)
        sets[name] = [expert_preds(s) for s in specs.split(",") if s]
    idx = next(iter(sets.values()))[0].index
    Y = Y.loc[idx]
    spec = DECOR.get(a.decor)
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(idx), len(idx)) for _ in range(500)]

    def macro(S, rows=None):
        yy, ss = (Y, S) if rows is None else (Y.iloc[rows], S.iloc[rows])
        return float(np.mean([roc_auc_score(yy[f], ss[f]) for f in LAB if 0 < yy[f].sum() < len(yy)]))

    rows, per, base = [], {}, None
    for name, P in sets.items():
        Z = zscores(P, idx)
        m = macro(Z)
        r = {"set": name, "models": len(P), "macro": round(m, 4), "dev": round(macro(Z, np.arange(29)), 4),
             "holdout": round(macro(Z, np.arange(29, len(idx))), 4)}
        if base is None:
            base = Z
        else:
            d = [macro(Z, b) - macro(base, b) for b in boots]
            r["vs first"] = f"{m - macro(base):+.4f} +-{np.std(d):.4f}"
        if spec:
            S = decor(Z, spec)
            d = [macro(S, b) - macro(Z, b) for b in boots]
            r[f"+{a.decor}"] = f"{macro(S):.4f} ({macro(S) - m:+.4f} +-{np.std(d):.4f})"
        r["MedOA~LatOA"] = round(Z["Medial OA"].corr(Z["Lateral OA"], method="spearman"), 2)
        r["Eff~Syn"] = round(Z["Effusion"].corr(Z["Synovitis"], method="spearman"), 2)
        r["MedMen~LatMen"] = round(Z["Medial Meniscus"].corr(Z["Lateral Meniscus"], method="spearman"), 2)
        rows.append(r)
        per[name] = {f: roc_auc_score(Y[f], Z[f]) for f in LAB}
    print(f"{len(idx)} experts (29 gold-dev + 29 holdout) | expert label phi: MedOA~LatOA 0.32, Eff~Syn 0.40, MedMen~LatMen 0.12")
    print(pd.DataFrame(rows).fillna("").to_string(index=False))
    print("\nper finding:")
    print(pd.DataFrame(per).round(3).to_string())


if __name__ == "__main__":
    main()
