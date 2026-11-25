#!/usr/bin/env python3
"""Targets for the external (fastMRI) studies, on the scale of the round labels, appended to a labels parquet.

    python scripts/make_ext_labels.py --labels labels_r3.parquet --out labels_r3x.parquet \
        --family c224hvy=outputs/runpod_results/r2-c224hvy40-f*/*:outputs/ext/raptor_ext_c224hvy.npz \
        --family effv2s=outputs/runpod_results/r2-effv2s336-f*/*:outputs/ext/raptor_ext_effv2s.npz ...

Each family = the result dirs of its fold models (their OOF predictions on competition studies) and the file written
by runpod/ext_pseudo.py with the SAME checkpoints scored on the external cache (per_ckpt, in --ckpts order).
Same rule as make_round_labels.py, minus the report (external studies have none):
  e   = mean over families of the mean over fold models of the external study's percentile inside that model's own
        held-out competition predictions (so a fastMRI study is placed on the competition scale, model by model);
  p   = Platt(e -> report label), fitted on the competition studies exactly as make_round_labels.py does;
  target(external) = p.
The reference predictions are the best-epoch OOF files (raptor_oof_<tag>_best_fold<k>.npz, fetched from the result
dataset when missing) because ext_pseudo.py scores the best-epoch checkpoint; the primary OOF file is the fallback.
"""
from __future__ import annotations

import argparse
import glob
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO = Path(__file__).resolve().parents[1]
KAGGLE = str(Path(sys.executable).with_name("kaggle"))
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def best_oof(d: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """(ids, pred, source) of one fold model's held-out competition predictions, best-epoch file preferred."""
    prim = [f for f in glob.glob(f"{d}/raptor_oof_*_fold[0-9].npz")
            if not re.search(r"_(swa|ema|last|best|epens|ep\d+)_fold\d+\.npz$", f)]
    assert len(prim) == 1, f"{d}: expected one primary OOF file, found {len(prim)}"
    tag, fold = re.search(r"raptor_oof_(.+)_fold(\d)\.npz$", prim[0]).groups()
    name = f"raptor_oof_{tag}_best_fold{fold}.npz"
    best = Path(d) / name
    if not best.exists():
        for path in (f"{tag}/{name}", name):
            subprocess.run([KAGGLE, "datasets", "download", f"girishbose/rsna-knee-rp-{Path(d).parent.name}-out", "-f", path,
                            "-p", str(d), "-q"], capture_output=True, text=True, timeout=600)
            z = Path(d) / (name + ".zip")
            if z.exists():
                subprocess.run(["unzip", "-o", "-q", str(z), "-d", str(d)])
                z.unlink()
            if best.exists():
                break
    src = best if best.exists() else Path(prim[0])
    z = np.load(src, allow_pickle=True)
    return z["ids"].astype(str), z["pred"].astype(np.float64), ("best" if best.exists() else "primary") + f":{tag}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="round labels parquet under data/processed (competition rows)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", action="append", required=True, help="name=<glob of fold result dirs>:<raptor_ext npz>")
    ap.add_argument("--report", default="labels_bestof_v1.parquet")
    a = ap.parse_args()
    rep = pd.read_parquet(REPO / "data" / "processed" / a.report)
    rep = rep.assign(StudyInstanceUID=rep["StudyInstanceUID"].astype(str)).set_index("StudyInstanceUID")[LAB].astype(float)
    Ec, Ee, ext_ids = [], [], None
    for spec in a.family:
        name, rest = spec.split("=", 1)
        pat, extf = rest.rsplit(":", 1)
        dirs = sorted(d for d in glob.glob(pat) if glob.glob(f"{d}/raptor_oof_*_fold[0-9].npz"))
        z = np.load(extf, allow_pickle=True)
        ck = [c.split(":")[1] for c in z["ckpts"].astype(str)]
        per = z["per_ckpt"].astype(np.float64)                      # (C, N_ext, 12)
        ids_e = z["ids"].astype(str)
        assert ext_ids is None or np.array_equal(ext_ids, ids_e), "external ids differ between families"
        ext_ids = ids_e
        comp_rank, ext_pct, used = [], [], []
        for d in dirs:
            ids, pred, src = best_oof(Path(d))
            tag = src.split(":")[1]
            if tag not in ck:
                raise SystemExit(f"{name}: checkpoint {tag} was not scored on the external cache ({extf})")
            comp_rank.append(pd.DataFrame(pred, index=ids, columns=LAB).rank(pct=True))
            srt = np.sort(pred, axis=0)
            e = per[ck.index(tag)]
            ext_pct.append(np.stack([np.searchsorted(srt[:, j], e[:, j], side="right") / len(srt) for j in range(12)], 1))
            used.append(src)
        print(f"{name}: {len(dirs)} fold models ({', '.join(used)}) | external {per.shape[1]} studies")
        Ec.append(pd.concat(comp_rank).groupby(level=0).mean())
        Ee.append(np.mean(ext_pct, axis=0))
    ids = sorted(set.intersection(*(set(e.index) for e in Ec)) & set(rep.index))
    E = sum(e.loc[ids] for e in Ec) / len(Ec)
    Eext = np.clip(sum(Ee) / len(Ee), 0.0, 1.0)
    T = pd.DataFrame(index=ext_ids, columns=LAB, dtype=float)
    print(f"\ncompetition studies for the fit: {len(ids)} | external: {len(ext_ids)}")
    print(f"{'finding':17s} prev(report)  P>=.5 comp  P>=.5 ext   mean P comp  mean P ext   median e ext")
    for j, f in enumerate(LAB):
        y = (rep.loc[ids, f] >= 0.5).astype(int)
        lr = LogisticRegression(C=10.0).fit(E[[f]].values, y)
        pc = lr.predict_proba(E[[f]].values)[:, 1]
        T[f] = lr.predict_proba(Eext[:, [j]])[:, 1]
        print(f"{f:17s} {y.mean():10.3f} {(pc >= .5).mean():10.3f} {(T[f] >= .5).mean():10.3f} {pc.mean():12.3f} {T[f].mean():11.3f} "
              f"{np.median(Eext[:, j]):12.3f}")
    base = pd.read_parquet(REPO / "data" / "processed" / a.labels)
    base["StudyInstanceUID"] = base["StudyInstanceUID"].astype(str)
    assert not base["StudyInstanceUID"].str.startswith("fm_").any(), "--labels already holds external rows"
    ext = T.reset_index().rename(columns={"index": "StudyInstanceUID"})
    out = pd.concat([base[["StudyInstanceUID", *LAB]], ext[["StudyInstanceUID", *LAB]]], ignore_index=True)
    out.to_parquet(REPO / "data" / "processed" / a.out, index=False)
    print(f"\nwrote data/processed/{a.out}: {len(base)} competition + {len(ext)} external rows")


if __name__ == "__main__":
    main()
