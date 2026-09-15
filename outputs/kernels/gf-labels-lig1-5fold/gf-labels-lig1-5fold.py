"""Train gf_labels_lig1 folds 0-4 on cache_gf_v0, then true OOF gold-58.

Teacher: weak_labels_gf_lig1.csv (weak_v1 + ACL/MCL/Med Men gap-fill).
Keep if OOF gold >= 0.6144 + 0.005.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

META = Path("/kaggle/input/datasets/girishbose/rsna-knee-gf-lig1-meta")
if not META.exists():
    META = Path("/kaggle/input/rsna-knee-gf-lig1-meta")

CACHE = Path("/kaggle/input/datasets/girishbose/rsna-knee-cache-gf-v0/cache_gf_v0")
if not CACHE.exists():
    CACHE = Path("/kaggle/input/rsna-knee-cache-gf-v0/cache_gf_v0")

TRAIN_CSV = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv")
OUT = Path("/kaggle/working/gf_labels_lig1_5fold")
N_FOLDS = 5
BASELINE_OOF_GOLD = 0.6144


def main() -> None:
    assert META.exists(), META
    assert CACHE.exists(), CACHE
    assert TRAIN_CSV.exists(), TRAIN_CSV
    weak = META / "data" / "processed" / "weak_labels_gf_lig1.csv"
    assert weak.exists(), weak
    print("cache npz:", len(list(CACHE.glob("*.npz"))), flush=True)

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
            "configs/gf_labels_lig1.yaml",
            "--train-csv",
            str(TRAIN_CSV),
            "--folds",
            "data/folds/folds_v1.csv",
            "--weak-csv",
            "data/processed/weak_labels_gf_lig1.csv",
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

    oof_parts = [pd.read_csv(OUT / f"fold{f}_oof.csv").assign(fold=f) for f in range(N_FOLDS)]
    oof = pd.concat(oof_parts, ignore_index=True)
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
    print("=== gf_labels_lig1 5-fold OOF vs weak(+expert) ===", flush=True)
    print("macro_auc:", weak_summary["macro_auc"], flush=True)
    (OUT / "oof_weak_metrics.json").write_text(json.dumps(weak_summary, indent=2))

    is_gold = train[LABEL_COLS].notna().all(axis=1)
    gold = train.loc[is_gold, ["StudyInstanceUID"] + LABEL_COLS]
    g = gold.merge(preds, on="StudyInstanceUID", how="inner")
    assert len(g) == len(gold)
    gold_summary = summarize_metrics(
        g[LABEL_COLS].to_numpy(dtype=float),
        g[[f"{c}__p" for c in LABEL_COLS]].to_numpy(dtype=float),
    )
    print("=== gf_labels_lig1 5-fold OOF vs full-58 gold ===", flush=True)
    print("macro_auc:", gold_summary["macro_auc"], flush=True)
    for k, v in gold_summary["per_label_auc"].items():
        print(f"  {k}: {v}", flush=True)

    g_out = g[["StudyInstanceUID"]].copy()
    for c in LABEL_COLS:
        g_out[c] = g[f"{c}__p"]
    g_out.to_csv(OUT / "oof_gold58_preds.csv", index=False)
    keep_thr = BASELINE_OOF_GOLD + 0.005
    (OUT / "oof_gold58_metrics.json").write_text(
        json.dumps(
            {
                "weak_oof_macro_auc": weak_summary["macro_auc"],
                "gold58_oof_macro_auc": gold_summary["macro_auc"],
                "per_label_auc": gold_summary["per_label_auc"],
                "baseline_oof_gold58": BASELINE_OOF_GOLD,
                "keep_threshold": keep_thr,
                "verdict": (
                    "KEEP"
                    if gold_summary["macro_auc"] >= keep_thr
                    else "KILL"
                ),
                "note": "True OOF gold; teacher=gf_lig1 gap-fill on weak_v1",
            },
            indent=2,
        )
    )
    print("keep_threshold:", keep_thr, "verdict file written", flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
