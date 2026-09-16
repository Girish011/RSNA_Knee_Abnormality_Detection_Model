#!/usr/bin/env python3
"""Measure seed-to-seed noise of the true OOF gold-58 ruler, and build a quieter one.

Takes the `oof_gold58_preds.csv` from two or more runs of the SAME recipe that
differ only in `train.seed`. Because the recipe is identical, every difference
between them is noise, so their spread is the resolution limit of the ruler.

Outputs:
- per-seed gold macro AUC and the sample sd across seeds
- per-label sd across seeds (where the per-label stories come from)
- the seed-AVERAGED OOF (mean probability per study) and its gold macro, which is
  the variance-reduced baseline future A/Bs should be judged against
- a data-driven keep margin, replacing the guessed 0.005

Example
-------
    python scripts/seed_variance_report.py \
        --preds outputs/kaggle_download/gf-baseline-v0-5fold/gf_baseline_v0_5fold/oof_gold58_preds.csv \
                outputs/kaggle_download/gf-v0-seed1337-5fold/gf_v0_seed1337_5fold/oof_gold58_preds.csv \
                outputs/kaggle_download/gf-v0-seed2024-5fold/gf_v0_seed2024_5fold/oof_gold58_preds.csv \
        --labels 42 1337 2024 --out docs/audit/gf_v0_seed_variance.json
"""

from __future__ import annotations

import argparse
import json
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


def _macro(y: np.ndarray, p: np.ndarray) -> float:
    per = [_safe_auc(y[:, i], p[:, i]) for i in range(y.shape[1])]
    finite = [v for v in per if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", type=Path, required=True)
    ap.add_argument("--labels", nargs="*", default=None, help="names for each preds file")
    ap.add_argument("--train-csv", type=Path, default=Path("/Users/girish11/Downloads/train.csv"))
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    names = [str(x) for x in (args.labels or range(len(args.preds)))]
    if len(names) != len(args.preds):
        raise SystemExit("--labels must match --preds in length")

    train_csv = args.train_csv if args.train_csv.exists() else _ROOT / "data" / "raw" / "train.csv"
    train = pd.read_csv(train_csv)
    gold = train.loc[train[LABEL_COLS].notna().all(axis=1), [ID_COL] + LABEL_COLS]

    merged = gold.copy()
    for nm, p in zip(names, args.preds, strict=True):
        df = pd.read_csv(p)[[ID_COL] + LABEL_COLS].rename(
            columns={c: f"{c}__{nm}" for c in LABEL_COLS}
        )
        merged = merged.merge(df, on=ID_COL)
    y = merged[LABEL_COLS].to_numpy(dtype=float)
    print(f"gold studies: {len(merged)}   runs: {', '.join(names)}")

    probs = {nm: merged[[f"{c}__{nm}" for c in LABEL_COLS]].to_numpy(dtype=float) for nm in names}
    macros = {nm: _macro(y, probs[nm]) for nm in names}

    print("\nper-run gold macro AUC:")
    for nm in names:
        print(f"  seed {nm:<6} {macros[nm]:.4f}")
    vals = np.asarray([macros[nm] for nm in names], dtype=float)
    sd = float(vals.std(ddof=1)) if vals.size > 1 else float("nan")
    print(f"\nmean {vals.mean():.4f}   sd {sd:.4f}   range {vals.max() - vals.min():.4f}")

    pairwise = [
        (a, b, macros[b] - macros[a])
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    ]
    if pairwise:
        print("\npairwise same-recipe deltas (all pure noise):")
        for a, b, d in pairwise:
            print(f"  seed {b} - seed {a}: {d:+.4f}")
        worst = max(abs(d) for _, _, d in pairwise)
        print(f"  largest same-recipe |delta|: {worst:.4f}")
    else:
        worst = float("nan")

    print("\nper-label AUC by run (sd across runs):")
    per_label_sd = {}
    for i, c in enumerate(LABEL_COLS):
        aucs = [_safe_auc(y[:, i], probs[nm][:, i]) for nm in names]
        s = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else float("nan")
        per_label_sd[c] = s
        cells = "  ".join(f"{a:.3f}" for a in aucs)
        print(f"  {c:<18} {cells}   sd {s:.3f}")

    stacked = np.mean([probs[nm] for nm in names], axis=0)
    macro_avg = _macro(y, stacked)
    print(f"\nseed-averaged OOF gold macro: {macro_avg:.4f}  (mean of single seeds {vals.mean():.4f})")

    rng = np.random.default_rng(args.seed)
    n = len(merged)
    boot = []
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        v = _macro(y[idx], stacked[idx])
        if np.isfinite(v):
            boot.append(v)
    b = np.asarray(boot)
    print(
        f"seed-averaged bootstrap: mean {b.mean():.4f}  "
        f"95% CI [{np.quantile(b, 0.025):.4f}, {np.quantile(b, 0.975):.4f}]  sd {b.std(ddof=1):.4f}"
    )

    # Pre-registered (2026-09-16): margin must clear same-recipe noise, never below 0.005.
    margin = max(0.005, 2.0 * sd) if np.isfinite(sd) else 0.005
    print("\n=== proposed ruler (pre-registered form) ===")
    print(f"baseline (seed-averaged OOF gold) : {macro_avg:.4f}")
    print(f"measured seed sd                  : {sd:.4f}")
    print(f"keep margin = max(0.005, 2*sd)    : {margin:.4f}")
    print(f"keep threshold                    : {macro_avg + margin:.4f}")

    payload = {
        "runs": names,
        "per_run_gold_macro": macros,
        "mean_gold_macro": float(vals.mean()),
        "seed_sd": sd,
        "range": float(vals.max() - vals.min()),
        "largest_same_recipe_abs_delta": worst,
        "per_label_sd": per_label_sd,
        "seed_averaged_gold_macro": macro_avg,
        "seed_averaged_bootstrap": {
            "mean": float(b.mean()),
            "lo": float(np.quantile(b, 0.025)),
            "hi": float(np.quantile(b, 0.975)),
            "sd": float(b.std(ddof=1)),
        },
        "proposed_margin": margin,
        "proposed_keep_threshold": macro_avg + margin,
        "note": (
            "All runs share one recipe and differ only in train.seed, so every "
            "difference here is noise. Margin rule pre-registered before results: "
            "max(0.005, 2*seed_sd)."
        ),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print("\nwrote", args.out)


if __name__ == "__main__":
    main()
