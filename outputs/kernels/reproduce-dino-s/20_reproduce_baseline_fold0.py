# Kaggle T4 — reproduce the retained cache_v1 DINOv2-S fold0 baseline.
# Three deterministic FP32 seeds quantify the large variance seen in prior runs.

import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
INPUT = Path("/kaggle/input")
CODE = INPUT / "datasets/girishbose/rsna-knee-code"
if not (CODE / "scripts/train_baseline_fold.py").exists():
    CODE = next(path for path in INPUT.rglob("scripts/train_baseline_fold.py")).parents[1]

comp = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
cache_dir = next(
    path
    for path in INPUT.rglob("cache_v1")
    if path.is_dir() and len(list(path.glob("*.npz"))) >= 4400
)
weights = next(INPUT.rglob("dinov2_vits14_pretrain.pth"))
common = [
    sys.executable,
    "-u",
    str(CODE / "scripts/train_baseline_fold.py"),
    "--config",
    str(CODE / "configs/baseline_dinov2_s.yaml"),
    "--train-csv",
    str(comp / "train.csv"),
    "--folds",
    str(CODE / "data/folds/folds_v1.csv"),
    "--cache-dir",
    str(cache_dir),
    "--weak-csv",
    str(CODE / "data/processed/weak_labels_v1.csv"),
    "--weights",
    str(weights),
    "--fold",
    "0",
    "--epochs",
    "5",
    "--freeze-epochs",
    "5",
    "--pos-weight",
    "1.0",
    "--model-type",
    "baseline",
    "--disable-amp",
]

for seed in (42, 43, 44):
    out_dir = Path(f"/kaggle/working/reproduce_dinov2_s_fold0/seed{seed}")
    command = common + ["--seed", str(seed), "--out-dir", str(out_dir)]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)
