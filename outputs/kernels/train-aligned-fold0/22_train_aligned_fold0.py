# Kaggle T4 — fold0 from cache_v1 DINOv2-S 224 features.
# Control: feature aggregator trained from scratch on weak_v1.
# Challenger: same recipe, initialized from report-aligned aggregator.
# Attach: competition, rsna-knee-code, girishbose/report-align-dinov2-s-cachev1
# Reports are not used here. Gate vs seed42 image baseline 0.742 / ensemble 0.745.

import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

INPUT = Path("/kaggle/input")
CODE = INPUT / "datasets/girishbose/rsna-knee-code"
if not (CODE / "scripts/train_baseline_fold.py").exists():
    CODE = next(path for path in INPUT.rglob("scripts/train_baseline_fold.py")).parents[1]
train_csv = next(
    path
    for path in INPUT.rglob("train.csv")
    if "rsna-knee" in str(path).lower() and (path.parent / "train_series.csv").exists()
)
feature_dir = next(
    path
    for path in INPUT.rglob("features_dinov2_s_224")
    if path.is_dir() and sum(1 for _ in path.glob("*.npz")) >= 4000
)
ckpt = next(INPUT.rglob("report_aligned_dinov2_s.pt"))
config = Path("/kaggle/working/align_finetune.yaml")
config.write_text(
    """
experiment:
  id: aligned_dinov2_s_fold0
  track: main
  notes: "Frozen DINOv2-S 224 features; optional report-alignment init"
data:
  image_size: 224
model:
  backbone: dinov2_vits14
  type: feature_baseline
  input_dim: 384
  hidden_dim: 384
  freeze_backbone_epochs: 5
  dropout: 0.1
  pretrained: true
train:
  n_folds: 5
  seed: 42
  epochs: 5
  batch_size: 8
  lr: 3.0e-4
  weight_decay: 0.05
  unfreeze_lr_mult: 0.1
  amp: false
  num_workers: 0
loss:
  type: masked_bce
  use_confidence: true
  pos_weight: 1.0
"""
)

common = [
    sys.executable,
    "-u",
    str(CODE / "scripts/train_baseline_fold.py"),
    "--config",
    str(config),
    "--train-csv",
    str(train_csv),
    "--folds",
    str(CODE / "data/folds/folds_v1.csv"),
    "--cache-dir",
    str(feature_dir),
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
    "feature_baseline",
    "--seed",
    "42",
    "--disable-amp",
]
print("features", feature_dir, "ckpt", ckpt, flush=True)

for name, init in (("scratch", None), ("aligned", ckpt)):
    out_dir = Path(f"/kaggle/working/fold0_dinov2_s_224_{name}")
    command = common + ["--out-dir", str(out_dir)]
    if init is not None:
        command.extend(["--init-checkpoint", str(init)])
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)
print("done", flush=True)
