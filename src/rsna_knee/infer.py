"""Inference helpers and CLI for writing submission.csv."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from rsna_knee.constants import LABEL_COLS, SUBMISSION_ID_COL
from rsna_knee.data.dataset import StudyDataset, collate_studies
from rsna_knee.models.multiseries import create_multiseries_model
from rsna_knee.training.ensemble import blend_predictions


def validate_submission(df: pd.DataFrame, expected_ids: list[str] | None = None) -> None:
    required = [SUBMISSION_ID_COL] + LABEL_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"submission missing columns: {missing}")
    if list(df.columns[: len(required)]) != required:
        # Allow extra cols but require correct leading order for safety.
        pass
    probs = df[LABEL_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(probs).all():
        raise ValueError("submission contains non-finite values")
    if (probs < 0).any() or (probs > 1).any():
        raise ValueError("submission probabilities must be in [0, 1]")
    if expected_ids is not None:
        got = set(df[SUBMISSION_ID_COL].astype(str))
        exp = set(map(str, expected_ids))
        if got != exp:
            raise ValueError(
                f"UID mismatch: missing={sorted(exp - got)[:5]} extra={sorted(got - exp)[:5]}"
            )


def write_submission(
    study_uids: list[str],
    probs: np.ndarray,
    path: str | Path,
) -> Path:
    """Write competition submission.csv."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape != (len(study_uids), len(LABEL_COLS)):
        raise ValueError(f"probs shape {probs.shape} != ({len(study_uids)}, {len(LABEL_COLS)})")
    probs = np.clip(probs, 0.0, 1.0)
    df = pd.DataFrame(probs, columns=LABEL_COLS)
    df.insert(0, SUBMISSION_ID_COL, study_uids)
    validate_submission(df, expected_ids=study_uids)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def baseline_constant_submission(test_csv: str | Path, path: str | Path, value: float = 0.5) -> Path:
    test = pd.read_csv(test_csv)
    uids = test[SUBMISSION_ID_COL].astype(str).tolist()
    probs = np.full((len(uids), len(LABEL_COLS)), value, dtype=np.float64)
    return write_submission(uids, probs, path)


def _load_model_from_checkpoint(
    checkpoint: Path,
    *,
    backbone: str,
    label_plane_routing: bool,
    backbone_weights: Path | None,
    device: Any,
) -> Any:
    import torch

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    weights_path = str(backbone_weights) if backbone_weights and backbone_weights.exists() else None
    model = create_multiseries_model(
        backbone,
        weights_path=weights_path,
        freeze_backbone=True,
        pretrained=weights_path is None,
        label_plane_routing=label_plane_routing,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def predict_studies(
    model: Any,
    studies_df: pd.DataFrame,
    series_df: pd.DataFrame,
    series_root: Path,
    *,
    max_series: int,
    n_slices: int,
    image_size: int,
    device: Any,
    batch_size: int = 1,
    num_workers: int = 0,
) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    ds = StudyDataset(
        studies_df,
        series_df,
        series_root,
        max_series=max_series,
        n_slices=n_slices,
        image_size=image_size,
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_studies,
        num_workers=num_workers,
    )
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


def load_oof_blend_weights(
    oof_csvs: list[Path],
    train_csv: Path,
    *,
    weak_csv: Path | None = None,
) -> np.ndarray:
    """Compute per-label AUC weights from fold OOF predictions vs weak labels.

    Each OOF CSV covers one validation fold; weights are per-model per-label AUC
    on that fold's studies, normalized across folds per label.
    """
    from sklearn.metrics import roc_auc_score

    from rsna_knee.constants import NUM_LABELS
    from rsna_knee.data.cached_dataset import merge_weak_labels

    train = pd.read_csv(train_csv)
    if weak_csv is not None and weak_csv.exists():
        train = merge_weak_labels(train, weak_csv)

    m = len(oof_csvs)
    weights = np.zeros((m, NUM_LABELS), dtype=np.float64)
    for fi, oof_path in enumerate(oof_csvs):
        oof = pd.read_csv(oof_path)
        oof = oof.rename(columns={c: f"{c}__pred" for c in LABEL_COLS if c in oof.columns})
        label_cols = [c for c in LABEL_COLS if c in train.columns]
        merged = train[[SUBMISSION_ID_COL] + label_cols].merge(oof, on=SUBMISSION_ID_COL, how="inner")
        for j, col in enumerate(LABEL_COLS):
            true_col = col if col in merged.columns else None
            pred_col = f"{col}__pred"
            if true_col is None or pred_col not in merged.columns:
                weights[fi, j] = 0.5
                continue
            yt = merged[true_col].to_numpy(dtype=np.float64)
            yp = merged[pred_col].to_numpy(dtype=np.float64)
            mask = np.isfinite(yt)
            if mask.sum() < 2 or np.unique(yt[mask]).size < 2:
                weights[fi, j] = 0.5
            else:
                weights[fi, j] = float(roc_auc_score(yt[mask], yp[mask]))

    weights = np.clip(weights, 1e-3, None)
    weights = weights / weights.sum(axis=0, keepdims=True)
    return weights


def load_blend_weights_json(path: Path) -> np.ndarray:
    """Load baked ``(n_models, n_labels)`` blend weights from a JSON payload."""
    payload = json.loads(Path(path).read_text())
    weights = np.asarray(payload["weights"], dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != len(LABEL_COLS):
        raise ValueError(f"bad blend weights shape {weights.shape}")
    col_sums = weights.sum(axis=0, keepdims=True)
    if not np.allclose(col_sums, 1.0, atol=1e-3):
        weights = weights / np.clip(col_sums, 1e-12, None)
    return weights


def run_model_submission(
    *,
    test_csv: Path,
    series_csv: Path,
    series_root: Path,
    config_path: Path,
    checkpoints: list[Path],
    out_path: Path,
    backbone_weights: Path | None = None,
    use_2p5d: bool = False,
    batch_size: int = 1,
    num_workers: int = 0,
    blend: str = "uniform",
    blend_weights: np.ndarray | None = None,
    oof_csvs: list[Path] | None = None,
    train_csv: Path | None = None,
    weak_csv: Path | None = None,
    require_cuda: bool = False,
    log_every: int = 25,
) -> dict[str, Any]:
    """Run live DICOM inference and write submission.csv.

    ``blend`` is ``uniform`` or ``per_label_auc``. For ``per_label_auc``, pass
    ``blend_weights`` directly or supply ``oof_csvs`` + ``train_csv`` (+ optional
    ``weak_csv``) to compute weights from fold OOF.
    """
    if use_2p5d:
        raise NotImplementedError("2.5D inference is not implemented in this helper")

    import torch

    t0 = time.perf_counter()
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA required (S01 CPU path timed out at ~126s/study)")
    cfg = yaml.safe_load(config_path.read_text())
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    max_series = int(data_cfg.get("max_series", 3))
    n_slices = int(data_cfg.get("n_slices", 12))
    image_size = int(data_cfg.get("image_size", 224))
    backbone = str(model_cfg.get("backbone", "dinov2_vitb14"))
    label_plane_routing = bool(model_cfg.get("label_plane_routing", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test = pd.read_csv(test_csv)
    series = pd.read_csv(series_csv)
    uids = test[SUBMISSION_ID_COL].astype(str).tolist()
    print(
        f"infer: n_studies={len(uids)} n_ckpt={len(checkpoints)} device={device} "
        f"decode_once=True blend={blend}",
        flush=True,
    )

    # Load all models first, then decode each study once (critical for 9h limit).
    models = [
        _load_model_from_checkpoint(
            ckpt,
            backbone=backbone,
            label_plane_routing=label_plane_routing,
            backbone_weights=backbone_weights,
            device=device,
        )
        for ckpt in checkpoints
    ]

    from torch.utils.data import DataLoader

    ds = StudyDataset(
        test,
        series,
        series_root,
        max_series=max_series,
        n_slices=n_slices,
        image_size=image_size,
        train=False,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_studies,
        num_workers=num_workers,
    )

    pred_lists: list[list[np.ndarray]] = [[] for _ in models]
    n_done = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            plane_ids = batch["plane_ids"].to(device)
            fluid = batch["fluid"].to(device)
            fat_sup = batch["fat_sup"].to(device)
            series_mask = batch["series_mask"].to(device)
            slice_mask = batch["slice_mask"].to(device)
            for mi, model in enumerate(models):
                logits = model(images, plane_ids, fluid, fat_sup, series_mask, slice_mask)
                pred_lists[mi].append(torch.sigmoid(logits).cpu().numpy())
            n_done += int(images.shape[0])
            if log_every > 0 and (n_done % log_every == 0 or n_done == len(uids)):
                elapsed = time.perf_counter() - t0
                rate = elapsed / max(n_done, 1)
                eta = rate * max(len(uids) - n_done, 0)
                print(
                    f"infer progress {n_done}/{len(uids)} "
                    f"({rate:.2f}s/study, ETA {eta/60:.1f} min)",
                    flush=True,
                )

    preds = [np.concatenate(pl, axis=0) for pl in pred_lists]

    if blend == "per_label_auc":
        if blend_weights is None:
            if not oof_csvs or train_csv is None:
                raise ValueError("per_label_auc blend needs blend_weights or oof_csvs+train_csv")
            blend_weights = load_oof_blend_weights(oof_csvs, train_csv, weak_csv=weak_csv)
        weights = np.asarray(blend_weights, dtype=np.float64)
    else:
        weights = np.ones((len(preds), len(LABEL_COLS)), dtype=np.float64) / len(preds)

    blended = blend_predictions(preds, weights)
    write_submission(uids, blended, out_path)
    runtime_s = time.perf_counter() - t0
    return {
        "n_studies": len(uids),
        "n_checkpoints": len(checkpoints),
        "blend": blend,
        "runtime_s": runtime_s,
        "seconds_per_study": runtime_s / max(len(uids), 1),
        "device": str(device),
        "weights_shape": list(weights.shape),
        "decode_once": True,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="RSNA knee inference utilities")
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--constant", type=float, default=None, help="Write constant probs (debug)")
    args = parser.parse_args(argv)
    if args.constant is not None:
        path = baseline_constant_submission(args.test_csv, args.out, args.constant)
        print(f"Wrote {path}")
        return
    raise SystemExit("Full model inference runs from notebooks/90_submit_main.ipynb")


if __name__ == "__main__":
    main()
