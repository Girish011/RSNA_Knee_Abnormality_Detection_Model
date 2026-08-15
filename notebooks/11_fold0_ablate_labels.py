# Kaggle GPU T4x2 — fold0 A/B only (do NOT loop 5 folds)
# Goal: isolate weak_v2 vs weak_v1 with pos_weight=1.0 (match winning v1 recipe)
# Attach: competition + rsna-knee-code (latest) + dinov2-vits14 + rsna-knee-cache-v1
# Accelerator: T4x2 is fine — we use ONE process on cuda:0 (ignore GPU1)
# Internet: ON if third_party/dinov2 missing; OFF if hubconf exists
# Save Version: fold0-ablate-v2-pw1

from pathlib import Path
import os
import subprocess
import sys

COMP = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
CODE = Path("/kaggle/input/datasets/girishbose/rsna-knee-code")
WEIGHTS = Path("/kaggle/input/datasets/girishbose/dinov2-vits14-rsna-knee/dinov2_vits14_pretrain.pth")
CACHE = Path("/kaggle/input/notebooks/girishbose/rsna-knee-cache-v1/cache_v1")
REPO, SRC = CODE, CODE / "src"

os.environ["PYTHONPATH"] = str(SRC)
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # one GPU only; T4x2 second card idle on purpose
dino = REPO / "third_party" / "dinov2"
if (dino / "hubconf.py").exists():
    os.environ["DINOV2_REPO"] = str(dino)

assert COMP.exists() and WEIGHTS.exists() and CACHE.exists()
weak_v1 = REPO / "data/processed/weak_labels_v1.csv"
weak_v2 = REPO / "data/processed/weak_labels_v2.csv"
folds = REPO / "data/folds/folds_v1.csv"
cfg = REPO / "configs/baseline_dinov2_s.yaml"
assert weak_v1.exists() and weak_v2.exists() and folds.exists() and cfg.exists()

runs = [
    ("v1_pw1", weak_v1),
    ("v2_pw1", weak_v2),
]
out_root = Path("/kaggle/working/fold0_ablate")
out_root.mkdir(parents=True, exist_ok=True)

for tag, weak in runs:
    out = out_root / tag
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", str(REPO / "scripts/train_baseline_fold.py"),
        "--config", str(cfg),
        "--train-csv", str(COMP / "train.csv"),
        "--folds", str(folds),
        "--cache-dir", str(CACHE),
        "--weak-csv", str(weak),
        "--weights", str(WEIGHTS),
        "--fold", "0",
        "--freeze-epochs", "5",
        "--epochs", "5",
        "--pos-weight", "1.0",
        "--out-dir", str(out),
        "--device", "cuda",
    ]
    print("\n====", tag, weak.name, "====", flush=True)
    print("RUN", " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONPATH": str(SRC), "CUDA_VISIBLE_DEVICES": "0"}
    if "DINOV2_REPO" in os.environ:
        env["DINOV2_REPO"] = os.environ["DINOV2_REPO"]
    subprocess.check_call(cmd, env=env)

print("done — compare best macro_auc to v1 fold0 floor 0.764")
print("expected print lines: studies with any label 2449 (v1) then 2749 (v2)")
