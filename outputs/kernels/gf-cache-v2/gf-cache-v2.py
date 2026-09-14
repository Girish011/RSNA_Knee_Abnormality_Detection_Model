"""Build gf_v2 cache: same series picks as v0, 12 slices × 336.

Writes /kaggle/working/cache_gf_v2/*.npz (4407 studies).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

META = Path("/kaggle/input/datasets/girishbose/rsna-knee-gf-v2-meta")
if not META.exists():
    META = Path("/kaggle/input/rsna-knee-gf-v2-meta")

COMP = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection")
SERIES_ROOT = COMP / "train_series"
TRAIN_CSV = COMP / "train.csv"
SERIES_CSV = COMP / "train_series.csv"
PICKS = META / "outputs" / "eda" / "gf_v0_series_picks.csv"
OUT = Path("/kaggle/working/cache_gf_v2")


def main() -> None:
    assert META.exists(), META
    assert SERIES_ROOT.exists(), SERIES_ROOT
    assert PICKS.exists(), PICKS
    print("picks:", PICKS)
    print("series root:", SERIES_ROOT)

    os.chdir(META)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(META / "src") + os.pathsep + env.get("PYTHONPATH", "")

    OUT.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/build_cache.py",
        "--train-csv",
        str(TRAIN_CSV),
        "--series-csv",
        str(SERIES_CSV),
        "--series-root",
        str(SERIES_ROOT),
        "--out-dir",
        str(OUT),
        "--picks-csv",
        str(PICKS),
        "--max-series",
        "3",
        "--n-slices",
        "12",
        "--image-size",
        "336",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, env=env)

    npz = list(OUT.glob("*.npz"))
    print("npz count:", len(npz))
    assert len(npz) >= 4300, f"too few cache files: {len(npz)}"

    import numpy as np

    sample = np.load(npz[0], allow_pickle=True)
    print("sample file:", npz[0].name)
    print("images shape:", sample["images"].shape)
    assert sample["images"].shape[1:] == (12, 336, 336), sample["images"].shape
    print("DONE cache_gf_v2")


if __name__ == "__main__":
    main()
