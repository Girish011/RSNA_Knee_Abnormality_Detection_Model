#!/usr/bin/env python3
"""Package rsna-knee-rank1-patch for Kaggle dataset upload."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE = [
    "src/rsna_knee/models/plane_routing.py",
    "src/rsna_knee/models/multiseries.py",
    "src/rsna_knee/data/dicom.py",
    "src/rsna_knee/training/loss.py",
    "src/rsna_knee/infer.py",
    "scripts/train_baseline_fold.py",
    "scripts/build_cache.py",
    "scripts/infer_ensemble.py",
    "configs/rank1_v6c.yaml",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-zip", type=Path, default=Path("outputs/kaggle_rank1_patch/rsna-knee-rank1-patch.zip"))
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--slug", default="girishbose/rsna-knee-rank1-patch")
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
        upload_dir = args.out_zip.parent / "rank1_patch_upload"
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        shutil.unpack_archive(args.out_zip, upload_dir)
        subprocess.check_call(
            [str(kaggle), "datasets", "version", "-p", str(upload_dir), "-m", "rank1 infer+plane routing refresh"]
        )
        print("uploaded", args.slug)


if __name__ == "__main__":
    main()
