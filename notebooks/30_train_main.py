# Kaggle GPU T4x2 — DINOv2-B fold0 on weak_v1, FULLY FROZEN backbone
# Attach: competition + rsna-knee-code + dinov2-vitb14-rsna-knee + rsna-knee-cache-v1
# Save Version: main-dinov2b-fold0-frozen-weak-v1
# Prior unfreeze collapsed 0.729→0.615 — do not unfreeze.

from pathlib import Path
import os
import subprocess
import sys

COMP = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
CODE = Path("/kaggle/input/datasets/girishbose/rsna-knee-code")
WEIGHTS = Path("/kaggle/input/datasets/girishbose/dinov2-vitb14-rsna-knee/dinov2_vitb14_pretrain.pth")
CACHE = Path("/kaggle/input/notebooks/girishbose/rsna-knee-cache-v1/cache_v1")
SRC = CODE / "src"

if not WEIGHTS.exists():
    hits = list(Path("/kaggle/input").rglob("dinov2_vitb14_pretrain.pth"))
    if hits:
        WEIGHTS = hits[0]
if not CACHE.exists():
    hits = [p for p in Path("/kaggle/input").rglob("cache_v1") if p.is_dir()]
    if hits:
        CACHE = hits[0]
if not COMP.exists():
    for h in Path("/kaggle/input").rglob("train.csv"):
        if "rsna-knee" in str(h) and (h.parent / "train_series.csv").exists():
            COMP = h.parent
            break

for name, p in [("COMP", COMP), ("CODE", CODE), ("WEIGHTS", WEIGHTS), ("CACHE", CACHE)]:
    print(name, p, p.exists())
missing = [n for n, p in [("COMP", COMP), ("CODE", CODE), ("WEIGHTS", WEIGHTS), ("CACHE", CACHE)] if not p.exists()]
assert not missing, f"Missing inputs: {missing}"

os.environ["PYTHONPATH"] = str(SRC)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dino = CODE / "third_party" / "dinov2"
if (dino / "hubconf.py").exists():
    os.environ["DINOV2_REPO"] = str(dino)

cfg = CODE / "configs/main_dinov2_b.yaml"
weak = CODE / "data/processed/weak_labels_v1.csv"
folds = CODE / "data/folds/folds_v1.csv"
assert weak.exists() and cfg.exists() and folds.exists()

out = Path("/kaggle/working/main_dinov2_b_v1_frozen/fold0")
out.mkdir(parents=True, exist_ok=True)
cmd = [
    sys.executable, "-u", str(CODE / "scripts/train_baseline_fold.py"),
    "--config", str(cfg),
    "--train-csv", str(COMP / "train.csv"),
    "--folds", str(folds),
    "--cache-dir", str(CACHE),
    "--weak-csv", str(weak),
    "--weights", str(WEIGHTS),
    "--fold", "0",
    "--epochs", "8",
    "--freeze-epochs", "8",  # never unfreeze
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
