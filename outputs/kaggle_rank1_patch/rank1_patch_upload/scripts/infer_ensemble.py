#!/usr/bin/env python3
"""Run cached inference and blend multiple fold checkpoints into submission.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
import yaml

from rsna_knee.constants import LABEL_COLS, SUBMISSION_ID_COL
from rsna_knee.data.cached_dataset import CachedStudyDataset
from rsna_knee.data.dataset import collate_studies
from rsna_knee.infer import write_submission
from rsna_knee.models.multiseries import create_multiseries_model
from rsna_knee.training.ensemble import blend_predictions, per_label_blend_weights


def _load_checkpoint(path: Path, backbone: str, label_plane_routing: bool, device):
    import torch

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = create_multiseries_model(
        backbone,
        freeze_backbone=True,
        pretrained=False,
        label_plane_routing=label_plane_routing,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def predict_cached(
    model,
    studies_df: pd.DataFrame,
    cache_dir: Path,
    device,
    batch_size: int = 1,
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    ds = CachedStudyDataset(studies_df, cache_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_studies, num_workers=0)
    probs = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["images"].to(device),
                batch["plane_ids"].to(device),
                batch["fluid"].to(device),
                batch["fat_sup"].to(device),
                batch["series_mask"].to(device),
                batch["slice_mask"].to(device),
            )
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Blend fold checkpoints on cached studies")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    ap.add_argument("--study-csv", type=Path, required=True, help="CSV with StudyInstanceUID column")
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("submission.csv"))
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--blend-weights",
        type=Path,
        default=None,
        help="Optional npy (M,12) weights; default uniform",
    )
    args = ap.parse_args()

    import torch

    cfg = yaml.safe_load(args.config.read_text())
    backbone = str(cfg["model"].get("backbone", "dinov2_vitb14"))
    label_plane_routing = bool(cfg.get("model", {}).get("label_plane_routing", False))
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    studies = pd.read_csv(args.study_csv)
    uids = studies[SUBMISSION_ID_COL].astype(str).tolist()
    keep = [uid for uid in uids if (args.cache_dir / f"{uid}.npz").exists()]
    missing = len(uids) - len(keep)
    if missing:
        print(f"warning: {missing} studies missing cache — skipped")
    df = studies[studies[SUBMISSION_ID_COL].astype(str).isin(keep)].reset_index(drop=True)

    preds = []
    for ckpt in args.checkpoints:
        print("load", ckpt)
        model = _load_checkpoint(ckpt, backbone, label_plane_routing, device)
        preds.append(predict_cached(model, df, args.cache_dir, device))

    if args.blend_weights and args.blend_weights.exists():
        weights = np.load(args.blend_weights)
    else:
        weights = np.ones((len(preds), len(LABEL_COLS)), dtype=np.float64) / len(preds)

    blended = blend_predictions(preds, weights)
    out_uids = df[SUBMISSION_ID_COL].astype(str).tolist()
    path = write_submission(out_uids, blended, args.out)
    print(f"wrote {path} ({len(out_uids)} studies, {len(args.checkpoints)} models)")


if __name__ == "__main__":
    main()
