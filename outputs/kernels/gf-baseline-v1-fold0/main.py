"""Train gf_baseline_v1 fold0 (frozen DINOv2-S, 24-slice cache) then score all 58 gold.

Writes under /kaggle/working/gf_baseline_v1/:
  fold0_best.pt, fold0_oof.csv, fold0_oof_probs.npy, fold0_history.json
  fold0_gold58_preds.csv, fold0_gold58_metrics.json
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
    # Last resort: search for any cache_gf_v1 folder under input
    hits = list(Path("/kaggle/input").rglob("cache_gf_v1"))
    CACHE = hits[0] if hits else Path("/kaggle/input/MISSING_cache_gf_v1")

TRAIN_CSV = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv")
OUT = Path("/kaggle/working/gf_baseline_v1")


def main() -> None:
    assert META.exists(), META
    assert CACHE.exists(), CACHE
    assert TRAIN_CSV.exists(), TRAIN_CSV
    print("cache npz:", len(list(CACHE.glob("*.npz"))))

    os.chdir(META)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(META / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["DINOV2_REPO"] = str(META / "third_party" / "dinov2")

    OUT.mkdir(parents=True, exist_ok=True)
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
        "0",
        "--epochs",
        "5",
        "--freeze-epochs",
        "5",
        "--out-dir",
        str(OUT),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, env=env)

    ckpt = OUT / "fold0_best.pt"
    assert ckpt.exists(), ckpt
    print("checkpoint ok:", ckpt, ckpt.stat().st_size)

    # --- full-58 gold score ---
    sys.path.insert(0, str(META / "src"))
    import numpy as np
    import pandas as pd
    import torch

    from rsna_knee.constants import LABEL_COLS
    from rsna_knee.data.cached_dataset import CachedStudyDataset
    from rsna_knee.data.dataset import collate_studies
    from rsna_knee.metrics import summarize_metrics
    from rsna_knee.models.multiseries import create_multiseries_model

    train = pd.read_csv(TRAIN_CSV)
    is_gold = train[LABEL_COLS].notna().all(axis=1)
    gold = train.loc[is_gold].copy()
    have = [(CACHE / f"{uid}.npz").exists() for uid in gold["StudyInstanceUID"].astype(str)]
    gold = gold.loc[have].reset_index(drop=True)
    print("gold with cache:", len(gold))

    ds = CachedStudyDataset(gold, CACHE)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_studies)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_multiseries_model(
        "dinov2_vits14", freeze_backbone=True, pretrained=True, dropout=0.1
    )
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    probs_list, y_list, uids = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["images"].to(device),
                batch["plane_ids"].to(device),
                batch["fluid"].to(device),
                batch["fat_sup"].to(device),
                batch["series_mask"].to(device),
                batch["slice_mask"].to(device),
            )
            probs_list.append(torch.sigmoid(logits).cpu().numpy())
            y_list.append(batch["labels"].numpy())
            uids.extend(batch["study_uid"])

    y = np.concatenate(y_list)
    p = np.concatenate(probs_list)
    summary = summarize_metrics(y, p)

    print("=== gf_baseline_v1 fold0 vs full-58 gold ===")
    print("macro_auc:", summary["macro_auc"])
    print("n_defined_labels:", summary["n_defined_labels"])
    for k, v in summary["per_label_auc"].items():
        print(f"  {k}: {v}")

    pd.DataFrame(
        {"StudyInstanceUID": uids, **{c: p[:, i] for i, c in enumerate(LABEL_COLS)}}
    ).to_csv(OUT / "fold0_gold58_preds.csv", index=False)
    (OUT / "fold0_gold58_metrics.json").write_text(
        json.dumps(
            {
                "weak_val_best_macro_auc_from_ckpt": float(state.get("macro_auc", float("nan"))),
                "gold58_macro_auc": summary["macro_auc"],
                "per_label_auc": summary["per_label_auc"],
                "baseline_v0_gold58": 0.7281,
                "keep_threshold": 0.7281 + 0.005,
            },
            indent=2,
        )
    )
    print("wrote", OUT)
    print("files:", sorted(x.name for x in OUT.iterdir()))


if __name__ == "__main__":
    main()
