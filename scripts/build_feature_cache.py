#!/usr/bin/env python3
"""Encode each cached slice once with a frozen public image backbone."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
from tqdm import tqdm

from rsna_knee.models.backbone import create_image_encoder


def _images_from_cache(data: np.lib.npyio.NpzFile, use_2p5d: bool) -> np.ndarray:
    if use_2p5d:
        if "images_2p5d" not in data.files:
            raise KeyError("Cache has no images_2p5d; rebuild it with --store-2p5d")
        return data["images_2p5d"].astype(np.float32) / 255.0
    gray = data["images"].astype(np.float32) / 255.0
    return np.repeat(gray[:, :, None], 3, axis=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen per-slice feature cache")
    parser.add_argument("--image-cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--backbone",
        choices=["dinov2_vits14", "dinov2_vitb14", "mri_core_vitb"],
        required=True,
    )
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--slice-batch-size", type=int, default=12)
    parser.add_argument("--use-2p5d", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("Feature extraction requires a CUDA Kaggle notebook")
    device = torch.device("cuda:0")
    encoder = create_image_encoder(
        args.backbone,
        weights_path=args.weights,
        freeze=True,
        pretrained=args.weights is None,
        image_size=args.image_size,
    ).to(device)
    encoder.eval()

    paths = sorted(args.image_cache_dir.glob("*.npz"))
    if args.limit > 0:
        paths = paths[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = []

    for path in tqdm(paths, desc=f"features:{args.backbone}"):
        out_path = args.out_dir / path.name
        if out_path.exists():
            manifest.append({"study_uid": path.stem, "skipped": True})
            continue
        data = np.load(path, allow_pickle=True)
        images = _images_from_cache(data, args.use_2p5d)
        s, n, c, h, w = images.shape
        flat = torch.from_numpy(images.reshape(s * n, c, h, w))
        if (h, w) != (args.image_size, args.image_size):
            flat = F.interpolate(
                flat,
                size=(args.image_size, args.image_size),
                mode="bilinear",
                align_corners=False,
            )
        chunks = []
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            for start in range(0, len(flat), args.slice_batch_size):
                chunks.append(
                    encoder(flat[start : start + args.slice_batch_size].to(device))
                    .float()
                    .cpu()
                    .numpy()
                )
        features = np.concatenate(chunks).reshape(s, n, -1).astype(np.float16)
        slice_positions = (
            data["slice_positions"].astype(np.float32)
            if "slice_positions" in data.files
            else np.broadcast_to(np.linspace(0.0, 1.0, n), (s, n)).copy()
        )
        np.savez_compressed(
            out_path,
            features=features,
            plane_ids=data["plane_ids"],
            fluid=data["fluid"],
            fat_sup=data["fat_sup"],
            series_mask=data["series_mask"],
            slice_mask=data["slice_mask"],
            slice_positions=slice_positions,
        )
        manifest.append(
            {
                "study_uid": path.stem,
                "feature_shape": list(features.shape),
                "backbone": args.backbone,
                "use_2p5d": args.use_2p5d,
            }
        )

    elapsed = time.perf_counter() - started
    summary = {
        "backbone": args.backbone,
        "image_size": args.image_size,
        "use_2p5d": args.use_2p5d,
        "studies": len(paths),
        "runtime_s": elapsed,
        "studies_per_second": len(paths) / max(elapsed, 1e-6),
        "manifest": manifest,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "manifest"}, indent=2))


if __name__ == "__main__":
    main()
