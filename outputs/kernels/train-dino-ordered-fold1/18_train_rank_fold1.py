# Kaggle T4 — fold1 validation of the fold0-discovered ordered hybrid.

import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BACKBONE = "dinov2_vits14"
INPUT = Path("/kaggle/input")
CODE = INPUT / "datasets/girishbose/rsna-knee-code"
if not (CODE / "scripts/train_baseline_fold.py").exists():
    CODE = next(path for path in INPUT.rglob("scripts/train_baseline_fold.py")).parents[1]
train_csv = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv")
cache_dir = next(path for path in INPUT.rglob("cache_3x12_384") if path.is_dir())
feature_dir = next(
    path for path in INPUT.rglob(f"features_{BACKBONE}_384") if path.is_dir()
)
out_dir = Path("/kaggle/working/fold1_dinov2_vits14_ordered")

command = [
    sys.executable,
    str(CODE / "scripts/train_baseline_fold.py"),
    "--config",
    str(CODE / "configs/label_query_dinov2_s_384.yaml"),
    "--train-csv",
    str(train_csv),
    "--folds",
    str(CODE / "data/folds/folds_v1.csv"),
    "--cache-dir",
    str(cache_dir),
    "--feature-dir",
    str(feature_dir),
    "--weak-csv",
    str(CODE / "data/processed/weak_labels_v1.csv"),
    "--fold",
    "1",
    "--epochs",
    "5",
    "--freeze-epochs",
    "5",
    "--pos-weight",
    "1.0",
    "--model-type",
    "ordered",
    "--seed",
    "42",
    "--out-dir",
    str(out_dir),
]
print(" ".join(command), flush=True)
subprocess.run(command, check=True)
