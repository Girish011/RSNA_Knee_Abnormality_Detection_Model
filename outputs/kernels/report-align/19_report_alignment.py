# Kaggle T4x2 + Internet ON — training-only image/report alignment.
# Frozen DINOv2-S on cache_v1 (224), then InfoNCE vs multilingual-e5-base.
# Reports and the text encoder are never used at inference.
# Do not attach MRI-CORE or 384 rank features.

import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

INPUT = Path("/kaggle/input")
CODE = INPUT / "datasets/girishbose/rsna-knee-code"
if not (CODE / "scripts/pretrain_report_alignment.py").exists():
    CODE = next(path for path in INPUT.rglob("scripts/pretrain_report_alignment.py")).parents[1]
sys.path.insert(0, str(CODE / "src"))

comp = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
if not (comp / "train.csv").exists():
    hits = list(INPUT.rglob("train.csv"))
    comp = next(p.parent for p in hits if "rsna-knee" in str(p).lower() and (p.parent / "train_series.csv").exists())
cache_dir = next(
    path
    for path in INPUT.rglob("cache_v1")
    if path.is_dir() and len(list(path.glob("*.npz"))) >= 4400
)
weights = next(INPUT.rglob("dinov2_vits14_pretrain.pth"))
feature_dir = Path("/kaggle/working/features_dinov2_s_224")
out = Path("/kaggle/working/report_aligned_dinov2_s.pt")

extract = [
    sys.executable,
    "-u",
    str(CODE / "scripts/build_feature_cache.py"),
    "--image-cache-dir",
    str(cache_dir),
    "--out-dir",
    str(feature_dir),
    "--backbone",
    "dinov2_vits14",
    "--weights",
    str(weights),
    "--image-size",
    "224",
]
print(" ".join(extract), flush=True)
subprocess.run(extract, check=True)

align = [
    sys.executable,
    "-u",
    str(CODE / "scripts/pretrain_report_alignment.py"),
    "--train-csv",
    str(comp / "train.csv"),
    "--feature-dir",
    str(feature_dir),
    "--out",
    str(out),
    "--input-dim",
    "384",
    "--hidden-dim",
    "384",
    "--epochs",
    "5",
    "--batch-size",
    "64",
    "--seed",
    "42",
]
print(" ".join(align), flush=True)
subprocess.run(align, check=True)
print("done", flush=True)
