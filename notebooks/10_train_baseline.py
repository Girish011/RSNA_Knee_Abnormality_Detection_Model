# Kaggle GPU — DINOv2-S 5-fold with weak_labels_v2 (label A/B vs ~0.719)
# Attach: competition + rsna-knee-code + dinov2-vits14 (public) + rsna-knee-cache-v1
# Save Version name: baseline-dinov2s-5fold-weak-v2

from pathlib import Path
import os
import subprocess
import sys

COMP = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
CODE = Path("/kaggle/input/datasets/girishbose/rsna-knee-code")
WEIGHTS = Path("/kaggle/input/datasets/girishbose/dinov2-vits14-rsna-knee/dinov2_vits14_pretrain.pth")
CACHE = Path("/kaggle/input/notebooks/girishbose/rsna-knee-cache-v1/cache_v1")
REPO = CODE  # package layout: repo root at dataset root
SRC = REPO / "src"
os.environ["PYTHONPATH"] = str(SRC)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(SRC))

assert COMP.exists(), COMP
assert WEIGHTS.exists(), WEIGHTS
assert CACHE.exists(), CACHE
weak = REPO / "data/processed/weak_labels_v2.csv"
folds = REPO / "data/folds/folds_v1.csv"
cfg = REPO / "configs/baseline_dinov2_s.yaml"
assert weak.exists(), "Re-upload code dataset with weak_labels_v2.csv"
assert folds.exists() and cfg.exists()

out_root = Path("/kaggle/working/baseline_dinov2_s_v2")
out_root.mkdir(parents=True, exist_ok=True)

for fold in range(5):
    out = out_root / f"fold{fold}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", str(REPO / "scripts/train_baseline_fold.py"),
        "--config", str(cfg),
        "--train-csv", str(COMP / "train.csv"),
        "--folds", str(folds),
        "--cache-dir", str(CACHE),
        "--weak-csv", str(weak),
        "--weights", str(WEIGHTS),
        "--fold", str(fold),
        "--freeze-epochs", "5",
        "--epochs", "5",
        "--out-dir", str(out),
    ]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, env={**os.environ, "PYTHONPATH": str(SRC)})
print("done", out_root)
