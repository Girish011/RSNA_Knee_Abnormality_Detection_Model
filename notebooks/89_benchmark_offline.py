# Offline dry run on train DICOMs to project hidden-test submission runtime.
# Internet OFF. Attach code, competition, MRI-CORE source/weights, and one checkpoint.

import json
import os
import sys
from pathlib import Path

import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
N_STUDIES = 200
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
CODE = INPUT / "datasets/girishbose/rsna-knee-code"
if not (CODE / "src").exists():
    CODE = next(path for path in INPUT.rglob("src/rsna_knee")).parents[1]
sys.path.insert(0, str(CODE / "src"))

from rsna_knee.infer import run_model_submission

train_csv = next(
    path
    for path in INPUT.rglob("train.csv")
    if "rsna-knee" in str(path).lower() and (path.parent / "train_series.csv").exists()
)
comp = train_csv.parent
dry_csv = WORK / "dry_test.csv"
pd.read_csv(train_csv).iloc[:N_STUDIES][["StudyInstanceUID"]].to_csv(dry_csv, index=False)
checkpoint = next(INPUT.rglob("fold0_best.pt"))
mri_weights = next(INPUT.rglob("MRI_CORE_vitb.pth"))
mri_repo = CODE / "third_party/mri_foundation"
if not mri_repo.exists():
    mri_repo = next(path for path in INPUT.rglob("mri_foundation") if path.is_dir())
os.environ["MRI_CORE_REPO"] = str(mri_repo)

summary = run_model_submission(
    test_csv=dry_csv,
    series_csv=comp / "train_series.csv",
    series_root=comp / "train_series",
    config_path=CODE / "configs/label_query_mri_core_384.yaml",
    checkpoints=[checkpoint],
    out_path=WORK / "dry_submission.csv",
    backbone_weights=mri_weights,
    batch_size=1,
    num_workers=2,
)
summary["projected_4407_runtime_s"] = float(summary["seconds_per_study"]) * 4407
summary["under_9h_at_4407"] = summary["projected_4407_runtime_s"] < 9 * 60 * 60
print(json.dumps(summary, indent=2), flush=True)
