"""gf_mri_core stage 1: frozen MRI-CORE ViT-B on cache_gf_v0, true 5-fold OOF gold.

Same volume recipe as gf_v0 (3×12×224 cache). Encoder upsamples to 384 and pools
768-d ViT-B features (SAM neck skipped — not in the public teacher checkpoint).

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

META = Path("/kaggle/input/datasets/girishbose/rsna-knee-gf-mri-core-meta")
if not META.exists():
    META = Path("/kaggle/input/rsna-knee-gf-mri-core-meta")

CACHE = Path("/kaggle/input/datasets/girishbose/rsna-knee-cache-gf-v0/cache_gf_v0")
if not CACHE.exists():
    CACHE = Path("/kaggle/input/rsna-knee-cache-gf-v0/cache_gf_v0")

WEIGHTS_CANDIDATES = [
    Path("/kaggle/input/datasets/girishbose/mri-core-vitb-rsna-knee/MRI_CORE_vitb.pth"),
    Path("/kaggle/input/mri-core-vitb-rsna-knee/MRI_CORE_vitb.pth"),
]
WEIGHTS = next((p for p in WEIGHTS_CANDIDATES if p.exists()), None)
if WEIGHTS is None:
    hits = list(Path("/kaggle/input").rglob("MRI_CORE_vitb.pth"))
    WEIGHTS = hits[0] if hits else Path("/kaggle/input/MISSING_MRI_CORE_vitb.pth")

OUT = Path("/kaggle/working/gf_mri_core_v0_seed42_5fold")
N_FOLDS = 5
V0_SEED42_GOLD = 0.6144
KILL_BELOW = V0_SEED42_GOLD - 0.02
PROMISE_ABOVE = V0_SEED42_GOLD + 0.02


def _resolve_train_csv(meta: Path) -> Path:
    """Competition mount is flaky; prefer competition, then meta-bundled copy."""
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv"),
        Path("/kaggle/input/rsna-knee-abnormality-detection/train.csv"),
        meta / "data" / "raw" / "train.csv",
        meta / "data" / "processed" / "train.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    hits = sorted(Path("/kaggle/input").rglob("train.csv"))
    # Prefer competition / largest (full labels) over tiny stubs.
    hits = sorted(hits, key=lambda p: p.stat().st_size, reverse=True)
    if hits:
        return hits[0]
    print("=== /kaggle/input tree (debug) ===", flush=True)
    root = Path("/kaggle/input")
    if root.exists():
        for child in sorted(root.iterdir()):
            print(" ", child, flush=True)
            try:
                for sub in sorted(child.iterdir())[:30]:
                    print("   ", sub.name, flush=True)
            except OSError as exc:
                print("    (listdir failed)", exc, flush=True)
    raise FileNotFoundError("train.csv not found under /kaggle/input or meta")


def main() -> None:
    assert META.exists(), META
    assert CACHE.exists(), CACHE
    assert WEIGHTS.exists(), WEIGHTS
    train_csv = _resolve_train_csv(META)
    print("train_csv:", train_csv, "bytes:", train_csv.stat().st_size, flush=True)
    # Ensure einops (MRI-CORE SAM import chain) is present.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "einops"])
    n_npz = len(list(CACHE.glob("*.npz")))
    print("cache:", CACHE, "npz:", n_npz, flush=True)
    print("weights:", WEIGHTS, "bytes:", WEIGHTS.stat().st_size, flush=True)
    assert n_npz > 4000, n_npz

    os.chdir(META)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(META / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["MRI_CORE_REPO"] = str(META / "third_party" / "mri_foundation")

    OUT.mkdir(parents=True, exist_ok=True)
    for fold in range(N_FOLDS):
        cmd = [
            sys.executable,
            "scripts/train_baseline_fold.py",
            "--config",
            "configs/gf_mri_core_v0.yaml",
            "--train-csv",
            str(train_csv),
            "--folds",
            "data/folds/folds_v1.csv",
            "--weak-csv",
            "data/processed/weak_labels_v1.csv",
            "--cache-dir",
            str(CACHE),
            "--weights",
            str(WEIGHTS),
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

    train = pd.read_csv(train_csv)
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
    print("=== gf_mri_core 5-fold OOF vs weak(+expert) ===", flush=True)
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
    print("=== gf_mri_core 5-fold OOF vs full-58 gold ===", flush=True)
    print("macro_auc:", gold_summary["macro_auc"], flush=True)
    for lab, auc in gold_summary["per_label_auc"].items():
        print(f"  {lab}: {auc}", flush=True)

    gold_macro = float(gold_summary["macro_auc"])
    if gold_macro < KILL_BELOW:
        verdict = "KILL"
    elif gold_macro >= PROMISE_ABOVE:
        verdict = "PROMISING"
    else:
        verdict = "INCONCLUSIVE"
    report = {
        "seed": 42,
        "backbone": "mri_core_vitb",
        "embed_dim": 768,
        "image_size_encode": 384,
        "cache": "cache_gf_v0",
        "weak_oof_macro_auc": float(weak_summary["macro_auc"]),
        "gold58_oof_macro_auc": gold_macro,
        "per_label_auc": gold_summary["per_label_auc"],
        "v0_seed42_gold": V0_SEED42_GOLD,
        "delta_vs_v0_seed42": gold_macro - V0_SEED42_GOLD,
        "kill_below": KILL_BELOW,
        "promise_above": PROMISE_ABOVE,
        "verdict": verdict,
        "note": "Stage1 gate only; multi-seed only if PROMISING.",
    }
    (OUT / "oof_gold58_metrics.json").write_text(json.dumps(report, indent=2))
    g[["StudyInstanceUID"] + [f"{c}__p" for c in LABEL_COLS]].to_csv(
        OUT / "oof_gold58_preds.csv", index=False
    )
    print(
        f"verdict={verdict} gold={gold_macro:.4f} "
        f"(KILL<{KILL_BELOW:.4f} PROMISING>={PROMISE_ABOVE:.4f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
