"""Ruler-noise replicate: gf_baseline_v0 5-fold OOF at seed __SEED__.

Purpose is measurement, not a new recipe. Everything comes from the same
`rsna-knee-gf-v0-meta` code/labels and the same `cache_gf_v0` as the original
seed-42 run (OOF gold 0.6144); the ONLY difference is `train.seed`. The trainer
reads the seed from the config, so we patch exactly that one line at runtime and
assert nothing else changed.

Combined with the seed-42 run, this gives the seed-to-seed sd of true OOF gold-58
macro AUC, which is what the keep margin should have been derived from all along.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SEED = __SEED__

META = Path("/kaggle/input/datasets/girishbose/rsna-knee-gf-v0-meta")
if not META.exists():
    META = Path("/kaggle/input/rsna-knee-gf-v0-meta")

CACHE = Path("/kaggle/input/datasets/girishbose/rsna-knee-cache-gf-v0/cache_gf_v0")
if not CACHE.exists():
    CACHE = Path("/kaggle/input/rsna-knee-cache-gf-v0/cache_gf_v0")

TRAIN_CSV = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv")
OUT = Path(f"/kaggle/working/gf_v0_seed{SEED}_5fold")
CONFIG = Path(f"/kaggle/working/gf_baseline_v0_seed{SEED}.yaml")
N_FOLDS = 5
SEED42_OOF_GOLD = 0.6144


def write_seed_config() -> None:
    """Copy the v0 config with only `seed:` changed, and prove it."""
    src = (META / "configs" / "gf_baseline_v0.yaml").read_text()
    lines = src.splitlines(keepends=True)
    patched, n_hit = [], 0
    for line in lines:
        if line.strip().startswith("seed:"):
            indent = line[: len(line) - len(line.lstrip())]
            patched.append(f"{indent}seed: {SEED}\n")
            n_hit += 1
        else:
            patched.append(line)
    assert n_hit == 1, f"expected exactly one seed line, found {n_hit}"
    out = "".join(patched)
    diff = [
        (a, b) for a, b in zip(lines, patched, strict=True) if a != b
    ]
    assert len(diff) == 1, f"patched more than the seed line: {diff}"
    print("config diff:", diff[0][0].strip(), "->", diff[0][1].strip(), flush=True)
    CONFIG.write_text(out)


def main() -> None:
    assert META.exists(), META
    assert CACHE.exists(), CACHE
    assert TRAIN_CSV.exists(), TRAIN_CSV
    print("cache npz:", len(list(CACHE.glob("*.npz"))), flush=True)
    print("SEED:", SEED, flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    write_seed_config()

    os.chdir(META)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(META / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["DINOV2_REPO"] = str(META / "third_party" / "dinov2")

    for fold in range(N_FOLDS):
        cmd = [
            sys.executable,
            "scripts/train_baseline_fold.py",
            "--config",
            str(CONFIG),
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

    oof_parts = [pd.read_csv(OUT / f"fold{f}_oof.csv").assign(fold=f) for f in range(N_FOLDS)]
    oof = pd.concat(oof_parts, ignore_index=True)
    assert oof["StudyInstanceUID"].is_unique
    oof.to_csv(OUT / "oof_all.csv", index=False)

    train = pd.read_csv(TRAIN_CSV)
    weak_df = pd.read_csv(META / "data" / "processed" / "weak_labels_v1.csv")
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
    print(f"=== gf_v0 seed{SEED} 5-fold OOF vs weak(+expert) ===", flush=True)
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
    print(f"=== gf_v0 seed{SEED} 5-fold OOF vs full-58 gold ===", flush=True)
    print("macro_auc:", gold_summary["macro_auc"], flush=True)
    for k, v in gold_summary["per_label_auc"].items():
        print(f"  {k}: {v}", flush=True)

    g_out = g[["StudyInstanceUID"]].copy()
    for c in LABEL_COLS:
        g_out[c] = g[f"{c}__p"]
    g_out.to_csv(OUT / "oof_gold58_preds.csv", index=False)

    delta = gold_summary["macro_auc"] - SEED42_OOF_GOLD
    (OUT / "oof_gold58_metrics.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "weak_oof_macro_auc": weak_summary["macro_auc"],
                "gold58_oof_macro_auc": gold_summary["macro_auc"],
                "per_label_auc": gold_summary["per_label_auc"],
                "seed42_oof_gold58": SEED42_OOF_GOLD,
                "delta_vs_seed42": delta,
                "note": (
                    "Ruler-noise replicate of gf_v0. Same code/labels/cache as seed 42; "
                    "only train.seed differs. delta_vs_seed42 is pure run-to-run noise "
                    "because the recipe is identical - it is NOT a keep/kill signal."
                ),
            },
            indent=2,
        )
    )
    print(f"delta vs seed42 ({SEED42_OOF_GOLD}): {delta:+.4f}  [pure noise]", flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
