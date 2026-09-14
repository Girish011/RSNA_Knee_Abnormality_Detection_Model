"""Train gf_baseline_v0 folds 0-4 (frozen DINOv2-S, seed 42), then score true OOF gold-58.

Uses existing cache_gf_v0 (3x12x224). Writes under /kaggle/working/gf_baseline_v0_5fold/:
  fold{k}_best.pt, fold{k}_oof.csv, fold{k}_history.json
  oof_all.csv, oof_weak_metrics.json, oof_gold58_metrics.json, oof_gold58_preds.csv
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

META = Path("/kaggle/input/datasets/girishbose/rsna-knee-gf-v0-meta")
if not META.exists():
    META = Path("/kaggle/input/rsna-knee-gf-v0-meta")

CACHE = Path("/kaggle/input/datasets/girishbose/rsna-knee-cache-gf-v0/cache_gf_v0")
if not CACHE.exists():
    CACHE = Path("/kaggle/input/rsna-knee-cache-gf-v0/cache_gf_v0")

TRAIN_CSV = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv")
OUT = Path("/kaggle/working/gf_baseline_v0_5fold")
N_FOLDS = 5


def main() -> None:
    assert META.exists(), META
    assert CACHE.exists(), CACHE
    assert TRAIN_CSV.exists(), TRAIN_CSV
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
            "configs/gf_baseline_v0.yaml",
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
        ckpt = OUT / f"fold{fold}_best.pt"
        oof_path = OUT / f"fold{fold}_oof.csv"
        assert ckpt.exists(), ckpt
        assert oof_path.exists(), oof_path
        print(f"fold {fold} ok size={ckpt.stat().st_size}", flush=True)

    sys.path.insert(0, str(META / "src"))
    import pandas as pd

    from rsna_knee.constants import LABEL_COLS
    from rsna_knee.metrics import summarize_metrics

    oof_parts = []
    for fold in range(N_FOLDS):
        part = pd.read_csv(OUT / f"fold{fold}_oof.csv")
        part["fold"] = fold
        oof_parts.append(part)
    oof = pd.concat(oof_parts, ignore_index=True)
    assert oof["StudyInstanceUID"].is_unique, "duplicate OOF study rows"
    oof.to_csv(OUT / "oof_all.csv", index=False)
    print("OOF studies:", len(oof), flush=True)

    train = pd.read_csv(TRAIN_CSV)
    weak = pd.read_csv(META / "data" / "processed" / "weak_labels_v1.csv")
    y_weak = train[["StudyInstanceUID"] + LABEL_COLS].copy()
    w = weak.set_index("StudyInstanceUID")
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
    y = mw[[f"{c}__y" for c in LABEL_COLS]].to_numpy(dtype=float)
    p = mw[[f"{c}__p" for c in LABEL_COLS]].to_numpy(dtype=float)
    weak_summary = summarize_metrics(y, p)
    print("=== gf_v0 5-fold OOF vs weak(+expert override) ===", flush=True)
    print("macro_auc:", weak_summary["macro_auc"], flush=True)
    (OUT / "oof_weak_metrics.json").write_text(json.dumps(weak_summary, indent=2))

    is_gold = train[LABEL_COLS].notna().all(axis=1)
    gold = train.loc[is_gold, ["StudyInstanceUID"] + LABEL_COLS].copy()
    g = gold.merge(preds, on="StudyInstanceUID", how="inner")
    print("gold with OOF pred:", len(g), "/", len(gold), flush=True)
    assert len(g) == len(gold), f"missing OOF for some gold: {len(g)}/{len(gold)}"
    y_g = g[LABEL_COLS].to_numpy(dtype=float)
    p_g = g[[f"{c}__p" for c in LABEL_COLS]].to_numpy(dtype=float)
    gold_summary = summarize_metrics(y_g, p_g)
    print("=== gf_v0 5-fold OOF vs full-58 gold ===", flush=True)
    print("macro_auc:", gold_summary["macro_auc"], flush=True)
    print("n_defined_labels:", gold_summary["n_defined_labels"], flush=True)
    for k, v in gold_summary["per_label_auc"].items():
        print(f"  {k}: {v}", flush=True)

    g_out = g[["StudyInstanceUID"]].copy()
    for c in LABEL_COLS:
        g_out[c] = g[f"{c}__p"]
    g_out.to_csv(OUT / "oof_gold58_preds.csv", index=False)
    (OUT / "oof_gold58_metrics.json").write_text(
        json.dumps(
            {
                "weak_oof_macro_auc": weak_summary["macro_auc"],
                "gold58_oof_macro_auc": gold_summary["macro_auc"],
                "per_label_auc": gold_summary["per_label_auc"],
                "n_gold": int(len(g)),
                "prior_fold0_model_on_all58_gold": 0.7281,
                "note": "True OOF: each gold study scored by the fold that held it out",
            },
            indent=2,
        )
    )
    print("wrote", OUT, flush=True)
    print("files:", sorted(x.name for x in OUT.iterdir()), flush=True)


if __name__ == "__main__":
    main()
