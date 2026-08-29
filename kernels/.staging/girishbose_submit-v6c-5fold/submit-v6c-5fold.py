# S01b — RSNA Knee 5-fold v6c ensemble submit (uniform mean, decode-once, GPU)
# S01 (ref 55851760) TIMED OUT on hidden test (~126s/study on CPU, 5× DICOM decode).
# Fix: GPU + decode each study once then run all 5 models; attach rank1-patch for infer.
# Attach: competition + rsna-knee-code + dinov2-vitb14-rsna-knee + rsna-knee-rank1-patch
#   kernel_sources: train-b-fold{0..4}-weak-v6c
# GPU T4, internet OFF.

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def locate(name, contains=None, want_dir=False):
    for p in INPUT.rglob(name):
        if want_dir and not p.is_dir():
            continue
        if contains and contains.lower() not in str(p).lower():
            continue
        return p
    return None


def locate_all_checkpoints() -> list[Path]:
    found: dict[int, Path] = {}
    for p in INPUT.rglob("fold*_best.pt"):
        name = p.name
        if not name.startswith("fold") or not name.endswith("_best.pt"):
            continue
        try:
            fold = int(name.replace("fold", "").replace("_best.pt", ""))
        except ValueError:
            continue
        if 0 <= fold <= 4:
            found[fold] = p
    if len(found) == 5:
        return [found[i] for i in range(5)]
    if found:
        return [found[k] for k in sorted(found)]
    return []


CODE = next(
    (p for p in [INPUT / "datasets/girishbose/rsna-knee-code", INPUT / "rsna-knee-code"] if (p / "src").exists()),
    None,
)
if CODE is None:
    hit = locate("train_baseline_fold.py")
    CODE = hit.parents[1] if hit else None

PATCH = next(
    (
        p
        for p in [
            INPUT / "datasets/girishbose/rsna-knee-rank1-patch",
            INPUT / "rsna-knee-rank1-patch",
        ]
        if (p / "src").exists() or (p / "src/rsna_knee/infer.py").exists()
    ),
    None,
)
if PATCH is None:
    hit = locate("infer.py", contains="rank1")
    PATCH = hit.parents[2] if hit else None

# Prefer patch infer (decode-once) over stale code dataset.
sys.path.insert(0, str(CODE / "src"))
if PATCH is not None and (PATCH / "src").exists():
    sys.path.insert(0, str(PATCH / "src"))
dino = CODE / "third_party" / "dinov2"
if (dino / "hubconf.py").exists():
    os.environ["DINOV2_REPO"] = str(dino)

from rsna_knee.infer import baseline_constant_submission, run_model_submission

test_csv = locate("test.csv", contains="rsna-knee")
comp = test_csv.parent
series_csv = comp / "test_series.csv"
if not series_csv.exists():
    series_csv = locate("test_series.csv")
series_root = comp / "test_series"
if not series_root.exists():
    series_root = locate("test_series", want_dir=True)

WEIGHTS = locate("dinov2_vitb14_pretrain.pth")
CHECKPOINTS = locate_all_checkpoints()

for n, p in [
    ("CODE", CODE),
    ("PATCH", PATCH),
    ("test_csv", test_csv),
    ("series_csv", series_csv),
    ("series_root", series_root),
    ("WEIGHTS", WEIGHTS),
]:
    print(n, p, flush=True)
print("CHECKPOINTS", len(CHECKPOINTS), "cuda?", flush=True)
import torch

print("torch.cuda.is_available()", torch.cuda.is_available(), flush=True)
for i, ck in enumerate(CHECKPOINTS):
    print(f"  fold{i}", ck, ck.stat().st_size if ck.exists() else 0, flush=True)

cfg = WORK / "infer_v6c_5fold.yaml"
cfg.write_text(
    "model:\n  type: baseline\n  backbone: dinov2_vitb14\n  dropout: 0.15\n"
    "data:\n  max_series: 3\n  n_slices: 12\n  image_size: 224\n"
)

try:
    if len(CHECKPOINTS) < 1:
        raise RuntimeError(f"Need >=1 checkpoint, found {len(CHECKPOINTS)}")
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
        blend="uniform",
    )
    summary["n_checkpoints"] = len(CHECKPOINTS)
    summary["experiment"] = "S01b_v6c_5fold_gpu_decode_once"
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
