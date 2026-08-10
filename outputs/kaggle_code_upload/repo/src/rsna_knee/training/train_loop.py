"""Minimal training loop entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, device, scaler=None) -> float:
    from rsna_knee.training.loss import masked_bce_with_logits

    model.train()
    total = 0.0
    n = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        images = batch["images"].to(device)
        # Flatten leading dims handled inside model.
        kwargs = {
            "plane_ids": batch["plane_ids"].to(device),
            "fluid": batch["fluid"].to(device),
            "fat_sup": batch["fat_sup"].to(device),
            "series_mask": batch["series_mask"].to(device),
            "slice_mask": batch["slice_mask"].to(device),
        }
        labels = batch["labels"].to(device)
        mask = batch["label_mask"].to(device)
        conf = batch["label_confidence"].to(device)

        if scaler is not None:
            import torch

            with torch.autocast(device_type=device.type if hasattr(device, "type") else "cuda"):
                logits = model(images, **kwargs)
                loss = masked_bce_with_logits(logits, labels, mask, conf)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images, **kwargs)
            loss = masked_bce_with_logits(logits, labels, mask, conf)
            loss.backward()
            optimizer.step()

        total += float(loss.detach().cpu())
        n += 1
    return total / max(n, 1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train RSNA knee model")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    print(f"Loaded config: {args.config}")
    print(
        "This entrypoint validates config wiring. Full Kaggle training should "
        "use notebooks/10_train_baseline.ipynb with competition data attached."
    )
    print("Keys:", sorted(cfg.keys()))


if __name__ == "__main__":
    main()
