"""gf_labels_lig3 stage 1: clean Med-Men-only gap-fill (no new studies), seed 42.

Teacher = weak_v1 + Med Men fills restricted to the 2449 studies that already have
>=1 weak_v1 label. Training-set size matches gf_v0 exactly (single-factor vs lig2).

Gate (pre-registered; NOT a macro keep/kill):
  Medial Meniscus gold OOF >= 0.566  (= v0 seed42 Med Men 0.486 + 0.08)
If gate fails: kill without seeds 1337/2024.
If gate passes: run those seeds next for seed-averaged macro vs 0.6173 + 0.0427.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

META = Path("/kaggle/input/datasets/girishbose/rsna-knee-gf-lig3-meta")
if not META.exists():
    META = Path("/kaggle/input/rsna-knee-gf-lig3-meta")

CACHE = Path("/kaggle/input/datasets/girishbose/rsna-knee-cache-gf-v0/cache_gf_v0")
if not CACHE.exists():
    CACHE = Path("/kaggle/input/rsna-knee-cache-gf-v0/cache_gf_v0")

TRAIN_CSV = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv")
OUT = Path("/kaggle/working/gf_labels_lig3_5fold")
N_FOLDS = 5
SEED42_V0_MED_MEN = 0.486
GATE_DELTA = 0.08
GATE_MIN = SEED42_V0_MED_MEN + GATE_DELTA  # 0.566
SEED_AVERAGED_BASELINE = 0.6173
MARGIN = 0.0427


def main() -> None:
    assert META.exists(), META
    assert CACHE.exists(), CACHE
    assert TRAIN_CSV.exists(), TRAIN_CSV
    weak = META / "data" / "processed" / "weak_labels_gf_lig3.csv"
    assert weak.exists(), weak
    print("cache npz:", len(list(CACHE.glob("*.npz"))), flush=True)
    print(f"stage1 gate: Med Men gold OOF >= {GATE_MIN:.3f}", flush=True)

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
            "configs/gf_labels_lig3.yaml",
            "--train-csv",
            str(TRAIN_CSV),
            "--folds",
            "data/folds/folds_v1.csv",
            "--weak-csv",
            "data/processed/weak_labels_gf_lig3.csv",
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
    print("=== gf_labels_lig3 5-fold OOF vs weak(+expert) ===", flush=True)
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
    print("=== gf_labels_lig3 5-fold OOF vs full-58 gold ===", flush=True)
    print("macro_auc:", gold_summary["macro_auc"], flush=True)
    for k, v in gold_summary["per_label_auc"].items():
        print(f"  {k}: {v}", flush=True)

    g_out = g[["StudyInstanceUID"]].copy()
    for c in LABEL_COLS:
        g_out[c] = g[f"{c}__p"]
    g_out.to_csv(OUT / "oof_gold58_preds.csv", index=False)

    med_men = float(gold_summary["per_label_auc"]["Medial Meniscus"])
    gate_pass = med_men >= GATE_MIN
    print(
        f"\n=== STAGE1 GATE: Med Men gold {med_men:.4f} "
        f"{'PASS' if gate_pass else 'FAIL'} (need >= {GATE_MIN:.3f}) ===",
        flush=True,
    )
    if gate_pass:
        print("Next: seeds 1337 + 2024 for seed-averaged macro keep/kill.", flush=True)
    else:
        print("Kill without multi-seed. Med Men signal was likely confounded by new studies.", flush=True)

    (OUT / "oof_gold58_metrics.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "weak_oof_macro_auc": weak_summary["macro_auc"],
                "gold58_oof_macro_auc": gold_summary["macro_auc"],
                "per_label_auc": gold_summary["per_label_auc"],
                "stage1_gate_label": "Medial Meniscus",
                "stage1_gate_min": GATE_MIN,
                "stage1_gate_value": med_men,
                "stage1_gate_pass": gate_pass,
                "seed_averaged_v0_baseline": SEED_AVERAGED_BASELINE,
                "margin": MARGIN,
                "note": (
                    "Single-factor Med Men fills (no new studies). Stage1 gate is per-label "
                    "Med Men gold, not macro keep/kill. Macro keep needs 3-seed average."
                ),
            },
            indent=2,
        )
    )
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
