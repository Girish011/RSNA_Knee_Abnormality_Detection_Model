#!/usr/bin/env python3
"""Paired study-level bootstrap A/B of two runs' true OOF gold-58 predictions.

Answers the question the plain macro number cannot: is a delta of a few
thousandths distinguishable from resampling noise on 58 gold studies?

Also reports deltas split by whether a label's teacher cells actually changed
between the two runs. Labels whose supervision is identical give a direct read of
run-to-run noise (training stochasticity + n=58 sampling), which is the honest
scale against which the keep margin has to be judged.

Example
-------
    python scripts/gold_oof_ab.py \
        --baseline outputs/kaggle_download/gf-baseline-v0-5fold/gf_baseline_v0_5fold/oof_gold58_preds.csv \
        --candidate outputs/kaggle_download/gf-labels-lig2-5fold/gf_labels_lig2_5fold/oof_gold58_preds.csv \
        --changed-labels "Medial Meniscus"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd

from rsna_knee.constants import LABEL_COLS
from rsna_knee.metrics import _safe_auc

ID_COL = "StudyInstanceUID"


def _load_gold(train_csv: Path) -> pd.DataFrame:
    train = pd.read_csv(train_csv)
    is_gold = train[LABEL_COLS].notna().all(axis=1)
    return train.loc[is_gold, [ID_COL] + LABEL_COLS].reset_index(drop=True)


def _macro(y: np.ndarray, p: np.ndarray) -> float:
    per = [_safe_auc(y[:, i], p[:, i]) for i in range(y.shape[1])]
    finite = [v for v in per if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--train-csv", type=Path, default=Path("/Users/girish11/Downloads/train.csv"))
    ap.add_argument("--changed-labels", nargs="*", default=[])
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train_csv = args.train_csv if args.train_csv.exists() else _ROOT / "data" / "raw" / "train.csv"
    gold = _load_gold(train_csv)
    base = pd.read_csv(args.baseline)
    cand = pd.read_csv(args.candidate)

    base = base[[ID_COL] + LABEL_COLS].rename(columns={c: f"{c}_b" for c in LABEL_COLS})
    cand = cand[[ID_COL] + LABEL_COLS].rename(columns={c: f"{c}_c" for c in LABEL_COLS})
    m = gold.merge(base, on=ID_COL).merge(cand, on=ID_COL)
    y = m[LABEL_COLS].to_numpy(dtype=float)
    pb = m[[f"{c}_b" for c in LABEL_COLS]].to_numpy(dtype=float)
    pc = m[[f"{c}_c" for c in LABEL_COLS]].to_numpy(dtype=float)
    n = len(m)

    mb, mc = _macro(y, pb), _macro(y, pc)
    print(f"gold studies: {n}")
    print(f"baseline macro : {mb:.4f}")
    print(f"candidate macro: {mc:.4f}")
    print(f"delta          : {mc - mb:+.4f}")

    rng = np.random.default_rng(args.seed)
    deltas, bases, cands = [], [], []
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        a, b = _macro(y[idx], pb[idx]), _macro(y[idx], pc[idx])
        if np.isfinite(a) and np.isfinite(b):
            bases.append(a)
            cands.append(b)
            deltas.append(b - a)
    d = np.asarray(deltas)
    print(
        f"\npaired bootstrap delta: mean {d.mean():+.4f}  "
        f"95% CI [{np.quantile(d, 0.025):+.4f}, {np.quantile(d, 0.975):+.4f}]  sd {d.std(ddof=1):.4f}"
    )
    print(f"P(candidate > baseline) = {float((d > 0).mean()):.3f}")
    print(
        f"baseline  bootstrap sd {np.asarray(bases).std(ddof=1):.4f}   "
        f"candidate bootstrap sd {np.asarray(cands).std(ddof=1):.4f}"
    )

    changed = set(args.changed_labels)
    unchanged_deltas = []
    print("\nper-label AUC (teacher cells changed?):")
    for i, c in enumerate(LABEL_COLS):
        a, b = _safe_auc(y[:, i], pb[:, i]), _safe_auc(y[:, i], pc[:, i])
        tag = "CHANGED" if c in changed else "same"
        print(f"  {c:<18} {a:.3f} -> {b:.3f}  ({b - a:+.3f})  [{tag}]")
        if c not in changed:
            unchanged_deltas.append(b - a)

    if unchanged_deltas:
        u = np.asarray(unchanged_deltas)
        print(
            f"\nlabels with IDENTICAL teacher cells (n={u.size}): "
            f"mean delta {u.mean():+.4f}, sd {u.std(ddof=1):.4f}, "
            f"max |delta| {np.abs(u).max():.3f}"
        )
        print(
            "implied macro noise from run-to-run variation alone: "
            f"~{u.std(ddof=1) / np.sqrt(len(LABEL_COLS)):.4f} "
            "(sd of unchanged-label deltas / sqrt(12))"
        )


if __name__ == "__main__":
    main()
