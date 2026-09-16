# Kaggle T4x2 (GPU not required) — cache_v3: proven 3×12 selection at 384px.
# Attach competition + rsna-knee-code. Save output as rsna-knee-cache-384-v1.

import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
INPUT = Path("/kaggle/input")
CODE = INPUT / "datasets/girishbose/rsna-knee-code"
if not (CODE / "scripts/build_cache.py").exists():
    CODE = next(path for path in INPUT.rglob("scripts/build_cache.py")).parents[1]

# The code dataset also contains metadata CSVs, so require the mounted DICOM
# directory when locating the competition input.
comp = next(
    path
    for path in INPUT.rglob("train_series")
    if path.is_dir() and any(path.rglob("*.dcm"))
).parent
train_csv = comp / "train.csv"
if not train_csv.exists() or not (comp / "train_series.csv").exists():
    raise FileNotFoundError(f"Incomplete competition input at {comp}")
print(f"Competition input: {comp}", flush=True)

command = [
    sys.executable,
    str(CODE / "scripts/build_cache.py"),
    "--train-csv",
    str(train_csv),
    "--series-csv",
    str(comp / "train_series.csv"),
    "--series-root",
    str(comp / "train_series"),
    "--out-dir",
    "/kaggle/working/cache_3x12_384",
    "--max-series",
    "3",
    "--n-slices",
    "12",
    "--image-size",
    "384",
]
print(" ".join(command), flush=True)
subprocess.run(command, check=True)
