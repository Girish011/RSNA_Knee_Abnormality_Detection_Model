# Kaggle GPU T4 — Rank1 v6c (plane routing + per-label pos_weight)
# Uses rsna-knee-rank1-patch shadowing stale code trainer/model.
import json
import os
import subprocess
import sys
from pathlib import Path

COMP = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
CODE = Path("/kaggle/input/datasets/girishbose/rsna-knee-code")
PATCH = Path("/kaggle/input/datasets/girishbose/rsna-knee-rank1-patch")
WEIGHTS = Path("/kaggle/input/datasets/girishbose/dinov2-vitb14-rsna-knee/dinov2_vitb14_pretrain.pth")
CACHE = Path("/kaggle/input/notebooks/girishbose/rsna-knee-cache-v1/cache_v1")
FOLD = 1


def _find(pattern, is_dir=False, must_contain=None):
    for p in Path("/kaggle/input").rglob(pattern):
        if is_dir and not p.is_dir():
            continue
        if must_contain and not (p / must_contain).exists():
            continue
        return p
    return None


if not CODE.exists():
    hit = _find("scripts/train_baseline_fold.py")
    CODE = hit.parents[1] if hit else CODE
if not PATCH.exists():
    PATCH = _find("rank1_v6c.yaml")
    PATCH = PATCH.parents[1] if PATCH else PATCH
if not WEIGHTS.exists():
    WEIGHTS = _find("dinov2_vitb14_pretrain.pth") or WEIGHTS
if not CACHE.exists():
    CACHE = _find("cache_v1", is_dir=True) or CACHE
if not COMP.exists():
    for h in Path("/kaggle/input").rglob("train.csv"):
        if "rsna-knee" in str(h).lower() and (h.parent / "train_series.csv").exists():
            COMP = h.parent
            break
WEAK = _find("weak_labels_v6c_candidate.csv")

for name, p in [("COMP", COMP), ("CODE", CODE), ("PATCH", PATCH), ("WEIGHTS", WEIGHTS), ("CACHE", CACHE), ("WEAK", WEAK)]:
    print(name, p, p.exists() if p else False, flush=True)
missing = [n for n, p in [("COMP", COMP), ("CODE", CODE), ("PATCH", PATCH), ("WEIGHTS", WEIGHTS), ("CACHE", CACHE), ("WEAK", WEAK)] if not p or not p.exists()]
assert not missing, f"Missing inputs: {missing}"

patch_src = PATCH / "src"
code_src = CODE / "src"
train_script = PATCH / "scripts/train_baseline_fold.py"
if not train_script.exists():
    train_script = CODE / "scripts/train_baseline_fold.py"

os.environ["PYTHONPATH"] = f"{patch_src}:{code_src}"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
dino = CODE / "third_party" / "dinov2"
if (dino / "hubconf.py").exists():
    os.environ["DINOV2_REPO"] = str(dino)

cfg = PATCH / "configs/rank1_v6c.yaml"
folds = CODE / "data/folds/folds_v1.csv"
out = Path(f"/kaggle/working/rank1_v6c_b/fold{FOLD}")
out.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable, "-u", str(train_script),
    "--config", str(cfg),
    "--train-csv", str(COMP / "train.csv"),
    "--folds", str(folds),
    "--cache-dir", str(CACHE),
    "--weak-csv", str(WEAK),
    "--weights", str(WEIGHTS),
    "--fold", str(FOLD),
    "--epochs", "12",
    "--freeze-epochs", "12",
    "--out-dir", str(out),
    "--device", "cuda",
]
print("RUN", " ".join(cmd), flush=True)
env = {**os.environ, "PYTHONPATH": os.environ["PYTHONPATH"], "CUDA_VISIBLE_DEVICES": "0"}
if "DINOV2_REPO" in os.environ:
    env["DINOV2_REPO"] = os.environ["DINOV2_REPO"]
subprocess.check_call(cmd, env=env)

hist = out / f"fold{FOLD}_history.json"
if hist.exists():
    rows = json.loads(hist.read_text())
    best = max((r.get("val_macro_auc") or -1) for r in rows)
    print(f"BEST fold{FOLD} val_macro_auc (rank1_v6c) = {best:.4f}", flush=True)
print("done", out, flush=True)
