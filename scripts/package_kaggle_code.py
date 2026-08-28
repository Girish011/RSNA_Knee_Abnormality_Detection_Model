#!/usr/bin/env python3
"""Package rsna-knee source for Kaggle dataset upload (rsna-knee-code refresh)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE = [
    "src/rsna_knee",
    "scripts/train_baseline_fold.py",
    "scripts/build_cache.py",
    "scripts/infer_ensemble.py",
    "scripts/oof_report.py",
    "configs/rank1_v6c.yaml",
    "configs/v6c_gce.yaml",
    "configs/main_dinov2_b.yaml",
    "configs/baseline_dinov2_s.yaml",
    "data/folds/folds_v1.csv",
    "third_party/dinov2",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-zip", type=Path, default=Path("outputs/kaggle_code_upload/rsna-knee-code.zip"))
    ap.add_argument("--upload", action="store_true", help="kaggle datasets version after zip")
    ap.add_argument("--slug", default="girishbose/rsna-knee-code")
    args = ap.parse_args()

    args.out_zip.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "bundle"
        staging.mkdir()
        for rel in INCLUDE:
            src = ROOT / rel
            if not src.exists():
                print("skip missing", rel)
                continue
            dst = staging / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

        with zipfile.ZipFile(args.out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in staging.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(staging))

    print("wrote", args.out_zip, "size", args.out_zip.stat().st_size)
    if args.upload:
        kaggle = Path.home() / ".local/bin/kaggle"
        if not kaggle.exists():
            raise SystemExit("kaggle CLI not found")
        subprocess.check_call([str(kaggle), "datasets", "version", "-p", str(staging.parent), "-m", "rank1 plane routing"])
        print("uploaded", args.slug)


if __name__ == "__main__":
    main()
