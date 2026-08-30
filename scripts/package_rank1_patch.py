#!/usr/bin/env python3
"""Package rsna-knee-rank1-patch for Kaggle dataset upload."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE = [
    "src/rsna_knee",  # full package — partial trees break imports
    "scripts/train_baseline_fold.py",
    "scripts/build_cache.py",
    "scripts/infer_ensemble.py",
    "configs/rank1_v6c.yaml",
    "configs/s02_v6c_blend_weights.json",
    "data/folds/folds_v1.csv",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-zip", type=Path, default=Path("outputs/kaggle_rank1_patch/rsna-knee-rank1-patch.zip"))
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--slug", default="girishbose/rsna-knee-rank1-patch")
    args = ap.parse_args()

    args.out_zip.parent.mkdir(parents=True, exist_ok=True)
    upload_dir = args.out_zip.parent / "rank1_patch_upload"
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True)

    for rel in INCLUDE:
        src = ROOT / rel
        if not src.exists():
            print("skip missing", rel)
            continue
        dst = upload_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            for p in dst.rglob("*"):
                if p.is_file():
                    os.utime(p, None)
        else:
            shutil.copy2(src, dst)
            os.utime(dst, None)

    with zipfile.ZipFile(args.out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in upload_dir.rglob("*"):
            if p.is_file() and p.name != "dataset-metadata.json":
                zf.write(p, p.relative_to(upload_dir))

    print("wrote", args.out_zip, "size", args.out_zip.stat().st_size)
    if args.upload:
        kaggle = Path.home() / ".local/bin/kaggle"
        if not kaggle.exists():
            raise SystemExit("kaggle CLI not found")
        meta = {
            "id": args.slug,
            "title": "rsna-knee-rank1-patch",
            "licenses": [{"name": "CC0-1.0"}],
        }
        (upload_dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
        subprocess.check_call(
            [
                str(kaggle),
                "datasets",
                "version",
                "-p",
                str(upload_dir),
                "-m",
                "decode-once infer + S02 baked blend weights",
                "--dir-mode",
                "zip",
            ]
        )
        print("uploaded", args.slug)


if __name__ == "__main__":
    main()
