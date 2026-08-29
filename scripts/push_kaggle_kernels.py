#!/usr/bin/env python3
"""Push Kaggle script kernels from kernels/*-metadata.json descriptors."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kernels"
KAGGLE = Path.home() / ".local/bin/kaggle"

DOCKER = (
    "gcr.io/kaggle-private-byod/python@"
    "sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
)


def push_one(meta_path: Path, *, dry_run: bool = False) -> None:
    meta = json.loads(meta_path.read_text())
    slug = meta["id"]
    code_file = meta["code_file"]
    src = KERNELS / code_file
    if not src.exists():
        raise FileNotFoundError(f"missing kernel script {src}")

    staging = KERNELS / ".staging" / slug.replace("/", "_")
    staging.mkdir(parents=True, exist_ok=True)
    dst = staging / code_file
    dst.write_text(src.read_text())

    kernel_meta = {**meta, "code_file": code_file}
    kernel_meta.setdefault("language", "python")
    kernel_meta.setdefault("kernel_type", "script")
    kernel_meta.setdefault("docker_image", DOCKER)
    kernel_meta.setdefault("machine_shape", "NvidiaTeslaT4")
    (staging / "kernel-metadata.json").write_text(json.dumps(kernel_meta, indent=2))

    cmd = [str(KAGGLE), "kernels", "push", str(staging)]
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.check_call(cmd)
    print("pushed", slug)


def main() -> None:
    ap = argparse.ArgumentParser(description="Push Kaggle kernels from metadata JSON")
    ap.add_argument(
        "targets",
        nargs="*",
        help="Metadata basename(s) e.g. train-b-fold0-rank1-v6c-metadata.json; default rank1+submit",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--push-all", action="store_true", help="Push every *-metadata.json")
    args = ap.parse_args()

    if not KAGGLE.exists():
        raise SystemExit(f"kaggle CLI not found at {KAGGLE}")

    if args.push_all:
        metas = sorted(KERNELS.glob("*-metadata.json"))
    elif args.targets:
        metas = [KERNELS / t if t.endswith(".json") else KERNELS / f"{t}-metadata.json" for t in args.targets]
    else:
        metas = sorted(KERNELS.glob("*rank1*-metadata.json")) + sorted(KERNELS.glob("submit-*-metadata.json"))

    for meta in metas:
        if not meta.exists():
            print("skip missing", meta, file=sys.stderr)
            continue
        push_one(meta, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
