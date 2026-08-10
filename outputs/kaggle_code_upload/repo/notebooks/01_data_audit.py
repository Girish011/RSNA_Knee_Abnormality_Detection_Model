# RSNA Knee — Data Audit (run on Kaggle with competition data attached)
#
# Goals:
# - Label availability / prevalence
# - Report language rough stats
# - Series counts, plane/fluid/fat distributions
# - Spot-check DICOM transfer syntaxes
#
# Export summary tables into docs/ or a Kaggle dataset version note.

import os
from pathlib import Path

import pandas as pd

DATA = Path("/kaggle/input/rsna-knee-abnormality-detection")
# Local override for dry runs:
if not DATA.exists():
    DATA = Path(os.environ.get("RSNA_KNEE_DATA", "data/raw"))

train_csv = DATA / "train.csv"
series_csv = DATA / "train_series.csv"

if train_csv.exists():
    train = pd.read_csv(train_csv)
    print("studies", len(train))
    print(train.head())
else:
    print("train.csv not found — attach competition data on Kaggle")

if series_csv.exists():
    series = pd.read_csv(series_csv)
    print("series", len(series))
    print(series["Anatomical_Plane"].value_counts(dropna=False))
    print(series.groupby(["Fluid_Sensitive", "Fat_Suppression"]).size())
else:
    print("train_series.csv not found")
