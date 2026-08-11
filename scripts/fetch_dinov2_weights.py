#!/usr/bin/env python3
"""Download Meta DINOv2 weights for offline Kaggle packaging."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

URLS = {
    "vits14": (
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
        "dinov2_vits14_pretrain.pth",
    ),
    "vitb14": (
        "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
        "dinov2_vitb14_pretrain.pth",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(URLS), default="vits14")
    args = parser.parse_args()
    url, fname = URLS[args.model]
    out_dir = Path("data/external/dinov2")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / fname
    if out_file.exists() and out_file.stat().st_size > 1_000_000:
        print(f"Already present: {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, out_file)
    print(f"Wrote {out_file} ({out_file.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
