"""Build study-level resized cache for fast training (Kaggle or local sample)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/build_cache.py` without pip install -e .
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
from tqdm import tqdm

from rsna_knee.data.dicom import load_series_volume, sample_slice_indices
from rsna_knee.data.series import rank_and_select_series


def build_study_cache(
    study_uid: str,
    series_df: pd.DataFrame,
    series_root: Path,
    out_path: Path,
    *,
    max_series: int = 3,
    n_slices: int = 12,
    image_size: int = 224,
) -> dict:
    """Write one npz per study: images uint8 (S,N,H,W) + metadata arrays."""
    choices = rank_and_select_series(series_df, study_uid, max_series=max_series)
    s, n, h = max_series, n_slices, image_size
    images = np.zeros((s, n, h, h), dtype=np.uint8)
    plane_ids = np.zeros((s,), dtype=np.int64)
    fluid = np.zeros((s,), dtype=np.uint8)
    fat = np.zeros((s,), dtype=np.uint8)
    series_mask = np.zeros((s,), dtype=np.uint8)
    slice_mask = np.zeros((s, n), dtype=np.uint8)
    series_uids: list[str] = []

    for i, ch in enumerate(choices[:s]):
        series_dir = series_root / study_uid / ch.series_uid
        vol = load_series_volume(series_dir, target_size=(h, h))
        idxs = sample_slice_indices(len(vol), n, center_bias=True)
        if not idxs:
            series_uids.append(ch.series_uid)
            continue
        sampled = vol[idxs]
        images[i, : len(sampled)] = np.clip(sampled * 255.0, 0, 255).astype(np.uint8)
        slice_mask[i, : len(sampled)] = 1
        plane_ids[i] = ch.plane_id
        fluid[i] = ch.fluid_sensitive
        fat[i] = ch.fat_suppression
        series_mask[i] = 1
        series_uids.append(ch.series_uid)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        images=images,
        plane_ids=plane_ids,
        fluid=fluid,
        fat_sup=fat,
        series_mask=series_mask,
        slice_mask=slice_mask,
        series_uids=np.asarray(series_uids, dtype=object),
    )
    return {
        "study_uid": study_uid,
        "path": str(out_path),
        "n_series_selected": int(series_mask.sum()),
        "n_slices": n,
        "image_size": h,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RSNA knee study cache")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--series-csv", type=Path, required=True)
    parser.add_argument("--series-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-series", type=int, default=3)
    parser.add_argument("--n-slices", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=0, help="Optional study cap for smoke tests")
    parser.add_argument("--study-list", type=Path, default=None, help="Optional txt of StudyInstanceUIDs")
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv)
    series = pd.read_csv(args.series_csv)
    uids = train["StudyInstanceUID"].astype(str).tolist()
    if args.study_list and args.study_list.exists():
        uids = [ln.strip() for ln in args.study_list.read_text().splitlines() if ln.strip()]
    if args.limit and args.limit > 0:
        uids = uids[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for uid in tqdm(uids, desc="cache"):
        out_path = args.out_dir / f"{uid}.npz"
        if out_path.exists():
            manifest.append({"study_uid": uid, "path": str(out_path), "skipped": True})
            continue
        try:
            entry = build_study_cache(
                uid,
                series,
                args.series_root,
                out_path,
                max_series=args.max_series,
                n_slices=args.n_slices,
                image_size=args.image_size,
            )
            manifest.append(entry)
        except Exception as e:  # noqa: BLE001 — keep building despite bad series
            manifest.append({"study_uid": uid, "error": str(e)[:200]})

    man_path = args.out_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    ok = sum(1 for m in manifest if "error" not in m)
    print(f"Done: {ok}/{len(manifest)} ok → {man_path}")


if __name__ == "__main__":
    main()
