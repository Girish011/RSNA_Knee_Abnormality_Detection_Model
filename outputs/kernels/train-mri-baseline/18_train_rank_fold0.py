# Kaggle T4x2 — controlled fold0 rank-model ablation, one GPU only.
# Attach competition, rsna-knee-code, cache_3x12_384, and one feature dataset.

import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BACKBONE = "mri_core_vitb"  # or "dinov2_vits14"
MODEL_TYPE = "feature_baseline"  # feature_baseline -> ordered -> label_query
PLANE_PRIOR_PENALTY = 0.0  # label_query: run 0.0 first, then 2.0

INPUT = Path("/kaggle/input")
CODE = INPUT / "datasets/girishbose/rsna-knee-code"
if not (CODE / "scripts/train_baseline_fold.py").exists():
    CODE = next(path for path in INPUT.rglob("scripts/train_baseline_fold.py")).parents[1]
train_csv = next(
    path
    for path in INPUT.rglob("train.csv")
    if "rsna-knee" in str(path).lower() and (path.parent / "train_series.csv").exists()
)
cache_dir = next(path for path in INPUT.rglob("cache_3x12_384") if path.is_dir())
feature_dir = next(
    path
    for path in INPUT.rglob(f"features_{BACKBONE}_384")
    if path.is_dir()
)
config_name = (
    "label_query_mri_core_384.yaml"
    if BACKBONE == "mri_core_vitb"
    else "label_query_dinov2_s_384.yaml"
)
out_dir = Path(f"/kaggle/working/fold0_{BACKBONE}_{MODEL_TYPE}")

command = [
    sys.executable,
    str(CODE / "scripts/train_baseline_fold.py"),
    "--config",
    str(CODE / "configs" / config_name),
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
    "0",
    "--epochs",
    "5",
    "--freeze-epochs",
    "5",
    "--pos-weight",
    "1.0",
    "--model-type",
    MODEL_TYPE,
    "--out-dir",
    str(out_dir),
]
if MODEL_TYPE == "label_query":
    command.extend(["--plane-prior-penalty", str(PLANE_PRIOR_PENALTY)])
print(" ".join(command), flush=True)
subprocess.run(command, check=True)
