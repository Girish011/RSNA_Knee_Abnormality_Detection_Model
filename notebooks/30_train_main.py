# Kaggle GPU — DINOv2-B main (after weak_v2 S A/B shows lift)
# Attach: competition + rsna-knee-code + dinov2-vitb14 (public) + cache
# Prefer cache_v2 (4×16) if available; else reuse cache_v1 for first B smoke.
# Save Version: main-dinov2b-fold0-weak-v2

from pathlib import Path
import os
import subprocess
import sys

COMP = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
CODE = Path("/kaggle/input/datasets/girishbose/rsna-knee-code")
WEIGHTS = Path("/kaggle/input/datasets/girishbose/dinov2-vitb14-rsna-knee/dinov2_vitb14_pretrain.pth")
CACHE = Path("/kaggle/input/notebooks/girishbose/rsna-knee-cache-v1/cache_v1")
SRC = CODE / "src"
os.environ["PYTHONPATH"] = str(SRC)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(SRC))

cfg = CODE / "configs/main_dinov2_b.yaml"
weak = CODE / "data/processed/weak_labels_v2.csv"
folds = CODE / "data/folds/folds_v1.csv"
assert WEIGHTS.exists(), "Upload public dinov2-vitb14-rsna-knee dataset first"
assert weak.exists()

out = Path("/kaggle/working/main_dinov2_b_v2/fold0")
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
    "--out-dir", str(out),
]
print("RUN", " ".join(cmd), flush=True)
subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": str(SRC)})
print("done", out)
