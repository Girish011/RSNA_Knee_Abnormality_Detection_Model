# S02 — RSNA Knee 5-fold v6c per-label AUC-weighted blend (GPU decode-once)
# Prefer baked weights in rank1-patch (configs/s02_v6c_blend_weights.json).
# Queue only after scored S01b LB ≥ 0.690. Never INPUT.rglob the DICOM tree.
# GPU T4, internet OFF. On Sep-5 push with enable_gpu=true (repo metadata).

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def first_existing(*cands: Path) -> Path | None:
    for p in cands:
        if p is not None and p.exists():
            return p
    return None


def locate_fold_checkpoints() -> list[Path]:
    found: dict[int, Path] = {}
    roots: list[Path] = []
    for base in [INPUT, INPUT / "datasets", INPUT / "kernels"]:
        if not base.exists():
            continue
        for p in base.iterdir():
            if p.is_dir() and "train-b-fold" in p.name:
                roots.append(p)
    for root in roots:
        for p in root.rglob("fold*_best.pt"):
            name = p.name
            if not (name.startswith("fold") and name.endswith("_best.pt")):
                continue
            try:
                fold = int(name[4 : -len("_best.pt")])
            except ValueError:
                continue
            if 0 <= fold <= 4:
                found[fold] = p
    if len(found) == 5:
        return [found[i] for i in range(5)]
    return [found[k] for k in sorted(found)]


CODE = first_existing(INPUT / "datasets/girishbose/rsna-knee-code", INPUT / "rsna-knee-code")
PATCH = first_existing(
    INPUT / "datasets/girishbose/rsna-knee-rank1-patch",
    INPUT / "rsna-knee-rank1-patch",
)
COMP = first_existing(
    INPUT / "rsna-knee-abnormality-detection",
    INPUT / "competitions/rsna-knee-abnormality-detection",
)
WEIGHTS = first_existing(
    INPUT / "dinov2-vitb14-rsna-knee" / "dinov2_vitb14_pretrain.pth",
    INPUT / "datasets/girishbose/dinov2-vitb14-rsna-knee" / "dinov2_vitb14_pretrain.pth",
)

if CODE is None or PATCH is None or COMP is None:
    raise SystemExit(f"FATAL missing inputs CODE={CODE} PATCH={PATCH} COMP={COMP}")

sys.path.insert(0, str(CODE / "src"))
sys.path.insert(0, str(PATCH / "src"))
dino = CODE / "third_party" / "dinov2"
if (dino / "hubconf.py").exists():
    os.environ["DINOV2_REPO"] = str(dino)

from rsna_knee.infer import (
    baseline_constant_submission,
    load_blend_weights_json,
    run_model_submission,
)
import torch

test_csv = COMP / "test.csv"
series_csv = COMP / "test_series.csv"
series_root = COMP / "test_series"
CHECKPOINTS = locate_fold_checkpoints()
BAKED = first_existing(PATCH / "configs" / "s02_v6c_blend_weights.json")

for n, p in [
    ("CODE", CODE),
    ("PATCH", PATCH),
    ("COMP", COMP),
    ("test_csv", test_csv),
    ("WEIGHTS", WEIGHTS),
    ("BAKED", BAKED),
]:
    print(n, p, "ok" if p is not None and Path(p).exists() else "MISSING", flush=True)
print("CHECKPOINTS", len(CHECKPOINTS), flush=True)
print("torch.cuda.is_available()", torch.cuda.is_available(), flush=True)

cfg = WORK / "infer_v6c_5fold.yaml"
cfg.write_text(
    "model:\n  type: baseline\n  backbone: dinov2_vitb14\n  dropout: 0.15\n"
    "data:\n  max_series: 3\n  n_slices: 12\n  image_size: 224\n"
)

try:
    if len(CHECKPOINTS) < 5:
        raise RuntimeError(f"Need 5 checkpoints, found {len(CHECKPOINTS)}")
    if not torch.cuda.is_available():
        raise RuntimeError("S02 requires GPU; S01 CPU path timed out at ~126s/study")
    if BAKED is None:
        raise RuntimeError("Missing baked S02 weights in rank1-patch")
    blend_weights = load_blend_weights_json(BAKED)
    summary = run_model_submission(
        test_csv=test_csv,
        series_csv=series_csv,
        series_root=series_root,
        config_path=cfg,
        checkpoints=CHECKPOINTS,
        out_path=WORK / "submission.csv",
        backbone_weights=WEIGHTS,
        use_2p5d=False,
        batch_size=1,
        num_workers=2,
        blend="per_label_auc",
        blend_weights=blend_weights,
        require_cuda=True,
        log_every=25,
    )
    summary["experiment"] = "S02_v6c_5fold_per_label_auc"
    summary["baked_weights"] = str(BAKED)
    print(json.dumps(summary, indent=2), flush=True)
    if float(summary["runtime_s"]) >= 9 * 60 * 60:
        raise RuntimeError("Exceeded 9h limit")
except Exception as e:
    print("model inference failed, writing constant fallback:", repr(e), flush=True)
    baseline_constant_submission(test_csv, WORK / "submission.csv", 0.5)

import pandas as pd

sub = pd.read_csv(WORK / "submission.csv")
print("submission shape", sub.shape, flush=True)
print(sub.head(3).to_string(), flush=True)
print("wrote", WORK / "submission.csv", flush=True)
