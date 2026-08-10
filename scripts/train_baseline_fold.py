#!/usr/bin/env python3
"""Train one fold of the DINOv2-S baseline on a study cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rsna_knee.constants import LABEL_COLS, NUM_LABELS
from rsna_knee.data.cached_dataset import CachedStudyDataset, attach_folds, merge_weak_labels
from rsna_knee.data.dataset import collate_studies
from rsna_knee.metrics import macro_auc, summarize_metrics
from rsna_knee.models.baseline import create_baseline_model
from rsna_knee.training.loss import masked_bce_with_logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/baseline_dinov2_s.yaml"))
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--weak-csv", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--expert-only", action="store_true", help="Train only on 58 expert studies")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader

    cfg = yaml.safe_load(args.config.read_text())
    epochs = args.epochs or int(cfg["train"]["epochs"])
    lr = float(cfg["train"]["lr"])
    batch_size = int(cfg["train"]["batch_size"])

    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    )
    print("device", device)

    train = pd.read_csv(args.train_csv)
    train = merge_weak_labels(train, args.weak_csv)
    train = attach_folds(train, args.folds)

    if args.expert_only:
        train = train[train[LABEL_COLS].notna().all(axis=1)].copy()
        print("expert-only studies", len(train))

    # Keep studies that have at least one supervised label
    has_any = train[LABEL_COLS].notna().any(axis=1)
    train = train.loc[has_any].copy()
    print("studies with any label", len(train))

    val_df = train[train["fold"] == args.fold].reset_index(drop=True)
    tr_df = train[train["fold"] != args.fold].reset_index(drop=True)
    print(f"fold {args.fold}: train={len(tr_df)} val={len(val_df)}")

    # Filter to cached studies only
    def _cached(df: pd.DataFrame) -> pd.DataFrame:
        keep = []
        for uid in df["StudyInstanceUID"].astype(str):
            if (args.cache_dir / f"{uid}.npz").exists():
                keep.append(True)
            else:
                keep.append(False)
        out = df.loc[keep].reset_index(drop=True)
        return out

    tr_df = _cached(tr_df)
    val_df = _cached(val_df)
    print(f"after cache filter: train={len(tr_df)} val={len(val_df)}")
    if len(tr_df) == 0 or len(val_df) == 0:
        raise SystemExit("No cached studies for this fold — build cache first")

    tr_ds = CachedStudyDataset(tr_df, args.cache_dir)
    va_ds = CachedStudyDataset(val_df, args.cache_dir)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_studies, num_workers=0)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_studies, num_workers=0)

    weights = str(args.weights) if args.weights else None
    model = create_baseline_model(weights_path=weights, freeze_backbone=True, pretrained=weights is None)
    model.to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.05)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    history = []

    for epoch in range(epochs):
        if epoch == int(cfg["model"].get("freeze_backbone_epochs", 1)):
            for p in model.encoder.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(model.parameters(), lr=lr * 0.5, weight_decay=0.05)
            print("unfroze backbone")

        model.train()
        losses = []
        for batch in tr_loader:
            opt.zero_grad(set_to_none=True)
            logits = model(
                batch["images"].to(device),
                batch["plane_ids"].to(device),
                batch["fluid"].to(device),
                batch["fat_sup"].to(device),
                batch["series_mask"].to(device),
                batch["slice_mask"].to(device),
            )
            loss = masked_bce_with_logits(
                logits,
                batch["labels"].to(device),
                batch["label_mask"].to(device),
                batch["label_confidence"].to(device),
            )
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        # Validation
        model.eval()
        probs_list, y_list, m_list = [], [], []
        with torch.no_grad():
            for batch in va_loader:
                logits = model(
                    batch["images"].to(device),
                    batch["plane_ids"].to(device),
                    batch["fluid"].to(device),
                    batch["fat_sup"].to(device),
                    batch["series_mask"].to(device),
                    batch["slice_mask"].to(device),
                )
                probs = torch.sigmoid(logits).cpu().numpy()
                probs_list.append(probs)
                y_list.append(batch["labels"].numpy())
                m_list.append(batch["label_mask"].numpy())
        y = np.concatenate(y_list)
        p = np.concatenate(probs_list)
        m = np.concatenate(m_list)
        # Mask unsupervised as NaN for metrics
        y_metric = y.copy()
        y_metric[m <= 0] = np.nan
        summary = summarize_metrics(y_metric, p)
        score = summary["macro_auc"]
        print(f"epoch {epoch}: loss={np.mean(losses):.4f} val_macro_auc={score}")
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_macro_auc": score})
        if np.isfinite(score) and score > best:
            best = score
            torch.save(
                {"model": model.state_dict(), "fold": args.fold, "macro_auc": score, "epoch": epoch},
                args.out_dir / f"fold{args.fold}_best.pt",
            )
            np.save(args.out_dir / f"fold{args.fold}_oof_probs.npy", p)
            val_df[["StudyInstanceUID"]].assign(**{c: p[:, i] for i, c in enumerate(LABEL_COLS)}).to_csv(
                args.out_dir / f"fold{args.fold}_oof.csv", index=False
            )

    (args.out_dir / f"fold{args.fold}_history.json").write_text(json.dumps(history, indent=2))
    print(f"best macro_auc={best} → {args.out_dir}")


if __name__ == "__main__":
    main()
