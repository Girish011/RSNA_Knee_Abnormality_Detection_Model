"""gf_v1 stage 1 (honest ruler): 24-slice volume, true 5-fold OOF gold, seed 42.

Prior fold0→all58 kill (0.7089) is VOID under the 2026-09-15 true-OOF ruler.
This retests the same cache_gf_v1 with the honest metric.

Gate vs same-seed gf_v0 (0.6144), pre-registered:
  KILL if gold < 0.6144 - 0.02 = 0.5944
  PROMISING if gold >= 0.6144 + 0.02 = 0.6344  -> seeds 1337+2024
  else INCONCLUSIVE -> kill (not worth multi-seed)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

META = Path("/kaggle/input/datasets/girishbose/rsna-knee-gf-v1-meta")
if not META.exists():
    META = Path("/kaggle/input/rsna-knee-gf-v1-meta")

CACHE_CANDIDATES = [
    Path("/kaggle/input/datasets/girishbose/rsna-knee-cache-gf-v1/cache_gf_v1"),
    Path("/kaggle/input/rsna-knee-cache-gf-v1/cache_gf_v1"),
    Path("/kaggle/input/girishbose-gf-cache-v1/cache_gf_v1"),
    Path("/kaggle/input/gf-cache-v1/cache_gf_v1"),
]
CACHE = next((p for p in CACHE_CANDIDATES if p.exists()), None)
if CACHE is None:
    hits = list(Path("/kaggle/input").rglob("cache_gf_v1"))
    CACHE = hits[0] if hits else Path("/kaggle/input/MISSING_cache_gf_v1")

TRAIN_CSV = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv")
OUT = Path("/kaggle/working/gf_v1_true_oof_seed42_5fold")
N_FOLDS = 5
V0_SEED42_GOLD = 0.6144
KILL_BELOW = V0_SEED42_GOLD - 0.02
PROMISE_ABOVE = V0_SEED42_GOLD + 0.02


def main() -> None:
    assert META.exists(), META
    assert CACHE.exists(), CACHE
    assert TRAIN_CSV.exists(), TRAIN_CSV
    weak = META / "data" / "processed" / "weak_labels_v1.csv"
    assert weak.exists(), weak
    n_npz = len(list(CACHE.glob("*.npz")))
    print("cache:", CACHE, "npz:", n_npz, flush=True)
    assert n_npz > 4000, n_npz

    os.chdir(META)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(META / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["DINOV2_REPO"] = str(META / "third_party" / "dinov2")

    OUT.mkdir(parents=True, exist_ok=True)
    for fold in range(N_FOLDS):
        cmd = [
            sys.executable,
            "scripts/train_baseline_fold.py",
            "--config",
            "configs/gf_baseline_v1.yaml",
            "--train-csv",
            str(TRAIN_CSV),
            "--folds",
            "data/folds/folds_v1.csv",
            "--weak-csv",
            "data/processed/weak_labels_v1.csv",
            "--cache-dir",
            str(CACHE),
            "--fold",
            str(fold),
            "--epochs",
            "5",
            "--freeze-epochs",
            "5",
            "--out-dir",
            str(OUT),
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.check_call(cmd, env=env)
        assert (OUT / f"fold{fold}_best.pt").exists()
        assert (OUT / f"fold{fold}_oof.csv").exists()
        print(f"fold {fold} ok", flush=True)

    sys.path.insert(0, str(META / "src"))
    import pandas as pd

    from rsna_knee.constants import LABEL_COLS
    from rsna_knee.metrics import summarize_metrics

    oof = pd.concat(
        [pd.read_csv(OUT / f"fold{f}_oof.csv").assign(fold=f) for f in range(N_FOLDS)],
        ignore_index=True,
    )
    assert oof["StudyInstanceUID"].is_unique
    oof.to_csv(OUT / "oof_all.csv", index=False)

    train = pd.read_csv(TRAIN_CSV)
    weak_df = pd.read_csv(weak)
    y_weak = train[["StudyInstanceUID"] + LABEL_COLS].copy()
    w = weak_df.set_index("StudyInstanceUID")
    for idx, row in y_weak.iterrows():
        uid = str(row["StudyInstanceUID"])
        if uid not in w.index:
            continue
        wr = w.loc[uid]
        for c in LABEL_COLS:
            if pd.isna(row[c]) and c in wr.index and pd.notna(wr[c]):
                y_weak.at[idx, c] = wr[c]

    preds = oof.rename(columns={c: f"{c}__p" for c in LABEL_COLS})
    labs = y_weak.rename(columns={c: f"{c}__y" for c in LABEL_COLS})
    mw = preds.merge(labs, on="StudyInstanceUID", how="inner")
    weak_summary = summarize_metrics(
        mw[[f"{c}__y" for c in LABEL_COLS]].to_numpy(dtype=float),
        mw[[f"{c}__p" for c in LABEL_COLS]].to_numpy(dtype=float),
    )
    print("=== gf_v1 true OOF vs weak ===", weak_summary["macro_auc"], flush=True)
    (OUT / "oof_weak_metrics.json").write_text(json.dumps(weak_summary, indent=2))

    gold = train.loc[train[LABEL_COLS].notna().all(axis=1), ["StudyInstanceUID"] + LABEL_COLS]
    g = gold.merge(preds, on="StudyInstanceUID", how="inner")
    assert len(g) == len(gold)
    gold_summary = summarize_metrics(
        g[LABEL_COLS].to_numpy(dtype=float),
        g[[f"{c}__p" for c in LABEL_COLS]].to_numpy(dtype=float),
    )
    print("=== gf_v1 true OOF vs gold-58 ===", gold_summary["macro_auc"], flush=True)
    for k, v in gold_summary["per_label_auc"].items():
        print(f"  {k}: {v}", flush=True)

    g_out = g[["StudyInstanceUID"]].copy()
    for c in LABEL_COLS:
        g_out[c] = g[f"{c}__p"]
    g_out.to_csv(OUT / "oof_gold58_preds.csv", index=False)

    gold_m = float(gold_summary["macro_auc"])
    delta = gold_m - V0_SEED42_GOLD
    if gold_m < KILL_BELOW:
        verdict = "KILL"
    elif gold_m >= PROMISE_ABOVE:
        verdict = "PROMISING"
    else:
        verdict = "INCONCLUSIVE"

    print(
        f"\n=== STAGE1 vs v0 seed42 {V0_SEED42_GOLD:.4f}: "
        f"gold {gold_m:.4f} ({delta:+.4f}) -> {verdict} ===",
        flush=True,
    )

    (OUT / "oof_gold58_metrics.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "n_slices": 24,
                "weak_oof_macro_auc": weak_summary["macro_auc"],
                "gold58_oof_macro_auc": gold_m,
                "per_label_auc": gold_summary["per_label_auc"],
                "v0_seed42_gold": V0_SEED42_GOLD,
                "delta_vs_v0_seed42": delta,
                "kill_below": KILL_BELOW,
                "promise_above": PROMISE_ABOVE,
                "verdict": verdict,
                "note": (
                    "Honest true-OOF retest of 24-slice volume. Prior fold0→all58 kill is void. "
                    "Stage1 gate only; multi-seed only if PROMISING."
                ),
            },
            indent=2,
        )
    )
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
