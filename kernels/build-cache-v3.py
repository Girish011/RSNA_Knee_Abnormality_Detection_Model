# Kaggle GPU — build cache_v3 (4×16, ACL sagittal bias) — long-running
# Save output as rsna-knee-cache-v3 for rank1 retrain.
import os
import subprocess
import sys
from pathlib import Path

COMP = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
CODE = Path("/kaggle/input/datasets/girishbose/rsna-knee-code")
PATCH = Path("/kaggle/input/datasets/girishbose/rsna-knee-rank1-patch")


def _find(pattern):
    for p in Path("/kaggle/input").rglob(pattern):
        return p
    return None


if not PATCH.exists():
    PATCH = CODE
build_script = PATCH / "scripts/build_cache.py"
if not build_script.exists():
    build_script = _find("build_cache.py")
    build_script = build_script if build_script else CODE / "scripts/build_cache.py"

if not COMP.exists():
    for h in Path("/kaggle/input").rglob("train.csv"):
        if (h.parent / "train_series.csv").exists():
            COMP = h.parent
            break

out = Path("/kaggle/working/cache_v3")
out.mkdir(parents=True, exist_ok=True)
cmd = [
    sys.executable, "-u", str(build_script),
    "--train-csv", str(COMP / "train.csv"),
    "--series-csv", str(COMP / "train_series.csv"),
    "--series-root", str(COMP / "train_series"),
    "--out-dir", str(out),
    "--max-series", "4",
    "--n-slices", "16",
    "--sagittal-acl-bias",
]
print("RUN", " ".join(cmd), flush=True)
os.environ["PYTHONPATH"] = str(PATCH / "src") + ":" + str(CODE / "src")
subprocess.check_call(cmd, env=os.environ)
print("done", len(list(out.glob("*.npz"))), "studies")
