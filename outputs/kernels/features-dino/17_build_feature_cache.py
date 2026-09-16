# Kaggle T4x2 — frozen feature extraction, one GPU only.
# Attach rsna-knee-code + rsna-knee-cache-384-v1 + selected public weights.
# Set BACKBONE to one value and save separate dataset versions.

import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
BACKBONE = "dinov2_vits14"
USE_2P5D = False

INPUT = Path("/kaggle/input")
CODE = INPUT / "datasets/girishbose/rsna-knee-code"
if not (CODE / "scripts/build_feature_cache.py").exists():
    CODE = next(path for path in INPUT.rglob("scripts/build_feature_cache.py")).parents[1]
cache_dir = next(path for path in INPUT.rglob("cache_3x12_384") if path.is_dir())

if BACKBONE == "mri_core_vitb":
    weights = next(INPUT.rglob("MRI_CORE_vitb.pth"))
    repo = CODE / "third_party/mri_foundation"
    if not repo.exists():
        repo = next(path for path in INPUT.rglob("mri_foundation") if path.is_dir())
    os.environ["MRI_CORE_REPO"] = str(repo)
else:
    weights = next(INPUT.rglob("dinov2_vits14_pretrain.pth"))

out_dir = Path(f"/kaggle/working/features_{BACKBONE}_384")
command = [
    sys.executable,
    str(CODE / "scripts/build_feature_cache.py"),
    "--image-cache-dir",
    str(cache_dir),
    "--out-dir",
    str(out_dir),
    "--backbone",
    BACKBONE,
    "--weights",
    str(weights),
    "--image-size",
    "378",
    "--slice-batch-size",
    "12",
]
if USE_2P5D:
    command.append("--use-2p5d")
print(" ".join(command), flush=True)
subprocess.run(command, check=True)
