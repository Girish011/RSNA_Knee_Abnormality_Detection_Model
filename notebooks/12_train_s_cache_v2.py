# Kaggle GPU T4x2 — DINOv2-S fold0 on cache_v2 + weak_v1, backbone FROZEN
# Settings: Accelerator = GPU T4 x2 (NOT CPU). Do NOT pip install torch.
# Attach:
#   competition rsna-knee-abnormality-detection
#   girishbose/rsna-knee-code
#   girishbose/dinov2-vits14-rsna-knee
#   girishbose/rsna-knee-cache-v2
# Save Version: baseline-dinov2s-fold0-cache-v2-weak-v1

from pathlib import Path
import os
import subprocess
import sys

import torch

print("python", sys.executable)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA torch not available. In this notebook: Settings → Accelerator → GPU T4 x2, "
        "then Save Version again. Do not pip install torch (that installs CPU torch)."
    )

COMP = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
CODE = Path("/kaggle/input/datasets/girishbose/rsna-knee-code")
WEIGHTS = Path("/kaggle/input/datasets/girishbose/dinov2-vits14-rsna-knee/dinov2_vits14_pretrain.pth")
CACHE = Path("/kaggle/input/notebooks/girishbose/rsna-knee-cache-v2/cache_v2")
SRC = CODE / "src"

if not WEIGHTS.exists():
    hits = list(Path("/kaggle/input").rglob("dinov2_vits14_pretrain.pth"))
    if hits:
        WEIGHTS = hits[0]
if not CACHE.exists():
    hits = [p for p in Path("/kaggle/input").rglob("cache_v2") if p.is_dir()]
    if hits:
        CACHE = hits[0]

print("COMP", COMP, COMP.exists())
print("CODE", CODE, CODE.exists())
print("WEIGHTS", WEIGHTS, WEIGHTS.exists())
print("CACHE", CACHE, CACHE.exists())
assert COMP.exists() and CODE.exists() and WEIGHTS.exists() and CACHE.exists()

os.environ["PYTHONPATH"] = str(SRC)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dino = CODE / "third_party" / "dinov2"
if (dino / "hubconf.py").exists():
    os.environ["DINOV2_REPO"] = str(dino)

cfg = CODE / "configs" / "baseline_dinov2_s.yaml"
weak = CODE / "data" / "processed" / "weak_labels_v1.csv"
folds = CODE / "data" / "folds" / "folds_v1.csv"
assert weak.exists() and cfg.exists() and folds.exists()

out = Path("/kaggle/working/baseline_dinov2_s_cache_v2/fold0")
out.mkdir(parents=True, exist_ok=True)
cmd = [
    sys.executable, "-u", str(CODE / "scripts" / "train_baseline_fold.py"),
    "--config", str(cfg),
    "--train-csv", str(COMP / "train.csv"),
    "--folds", str(folds),
    "--cache-dir", str(CACHE),
    "--weak-csv", str(weak),
    "--weights", str(WEIGHTS),
    "--fold", "0",
    "--epochs", "5",
    "--freeze-epochs", "5",
    "--pos-weight", "1.0",
    "--out-dir", str(out),
    "--device", "cuda",
]
print("RUN", " ".join(cmd), flush=True)
env = {**os.environ, "PYTHONPATH": str(SRC), "CUDA_VISIBLE_DEVICES": "0"}
if "DINOV2_REPO" in os.environ:
    env["DINOV2_REPO"] = os.environ["DINOV2_REPO"]
subprocess.check_call(cmd, env=env)
print("done", out)
