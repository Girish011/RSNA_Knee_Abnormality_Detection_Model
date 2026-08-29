# S02 — RSNA Knee 5-fold v6c per-label AUC-weighted blend
# Hypothesis: weight each fold by its per-label val AUC (from OOF) beats uniform S01.
# Attach: competition + rsna-knee-code + dinov2-vitb14-rsna-knee + rsna-knee-weak-v6c
#   kernel_sources: train-b-fold{0..4}-weak-v6c (fold{F}_best.pt + fold{F}_oof.csv each)
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


def locate_checkpoints_and_oof() -> tuple[list[Path], list[Path]]:
    ckpts: dict[int, Path] = {}
    oofs: dict[int, Path] = {}
    for p in INPUT.rglob("fold*_best.pt"):
        try:
            fold = int(p.name.replace("fold", "").replace("_best.pt", ""))
        except ValueError:
            continue
        if 0 <= fold <= 4:
            ckpts[fold] = p
    for p in INPUT.rglob("fold*_oof.csv"):
        try:
            fold = int(p.name.replace("fold", "").replace("_oof.csv", ""))
        except ValueError:
            continue
        if 0 <= fold <= 4:
            oofs[fold] = p
    folds = sorted(set(ckpts) & set(oofs))
    return [ckpts[i] for i in folds], [oofs[i] for i in folds]


CODE = next(
    (p for p in [INPUT / "datasets/girishbose/rsna-knee-code", INPUT / "rsna-knee-code"] if (p / "src").exists()),
    None,
)
if CODE is None:
    hit = locate("train_baseline_fold.py")
    CODE = hit.parents[1] if hit else None
sys.path.insert(0, str(CODE / "src"))
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

train_csv = comp / "train.csv"
if not train_csv.exists():
    train_csv = locate("train.csv", contains="rsna-knee")
WEAK = locate("weak_labels_v6c_candidate.csv")
WEIGHTS = locate("dinov2_vitb14_pretrain.pth")
CHECKPOINTS, OOF_CSVS = locate_checkpoints_and_oof()

for n, p in [
    ("CODE", CODE),
    ("test_csv", test_csv),
    ("series_csv", series_csv),
    ("series_root", series_root),
    ("train_csv", train_csv),
    ("WEAK", WEAK),
    ("WEIGHTS", WEIGHTS),
]:
    print(n, p, flush=True)
print("CHECKPOINTS", len(CHECKPOINTS), "OOF_CSVS", len(OOF_CSVS), flush=True)

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
        blend="per_label_auc",
        oof_csvs=OOF_CSVS if OOF_CSVS else None,
        train_csv=train_csv,
        weak_csv=WEAK,
    )
    summary["experiment"] = "S02_v6c_5fold_per_label_auc"
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
