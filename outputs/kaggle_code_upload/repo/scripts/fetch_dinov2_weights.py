#!/usr/bin/env python3
"""Download Meta DINOv2-S weights for offline Kaggle packaging.

Saves to data/external/dinov2/ (gitignored). Upload that folder as a public
Kaggle Dataset so training/submit notebooks can run with internet off.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"
OUT_DIR = Path("data/external/dinov2")
OUT_FILE = OUT_DIR / "dinov2_vits14_pretrain.pth"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists() and OUT_FILE.stat().st_size > 1_000_000:
        print(f"Already present: {OUT_FILE} ({OUT_FILE.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, OUT_FILE)
    print(f"Wrote {OUT_FILE} ({OUT_FILE.stat().st_size / 1e6:.1f} MB)")
    print("Next: upload data/external/dinov2 as a public Kaggle Dataset named e.g. dinov2-vits14-rsna-knee")


if __name__ == "__main__":
    main()
