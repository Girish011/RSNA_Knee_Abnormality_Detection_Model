#!/usr/bin/env python3
"""Training-only multilingual image-report alignment over frozen slice features.

Uses the retained cache_v1 DINOv2-S aggregator, not the killed MRI-CORE/label-query path.
Reports and the text encoder are never used at inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd

from rsna_knee.data.cached_dataset import CachedFeatureDataset
from rsna_knee.data.dataset import collate_studies
from rsna_knee.models.multiseries import create_feature_multiseries_model


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def build_report_embeddings(train: pd.DataFrame, model_id: str, device, batch_size: int):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    text_model = AutoModel.from_pretrained(model_id).to(device).eval()
    texts = [
        "passage: " + " ".join(str(report if pd.notna(report) else "").split())
        for report in train["Report"]
    ]
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            output = text_model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            chunks.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
    embeddings = torch.cat(chunks)
    del text_model
    torch.cuda.empty_cache()
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Align MRI study features with reports")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--input-dim", type=int, default=384)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--text-model",
        default="intfloat/multilingual-e5-base",
        help="Public training-only multilingual report encoder",
    )
    args = parser.parse_args()

    import random

    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Run report alignment on a Kaggle GPU with Internet ON")
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    train = pd.read_csv(args.train_csv)
    train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)
    train = train[
        train["StudyInstanceUID"].map(lambda uid: (args.feature_dir / f"{uid}.npz").exists())
    ].reset_index(drop=True)
    print("studies", len(train), "text_model", args.text_model, flush=True)
    report_embeddings = build_report_embeddings(
        train, args.text_model, device, args.batch_size
    )
    uid_to_row = {uid: idx for idx, uid in enumerate(train["StudyInstanceUID"].tolist())}

    dataset = CachedFeatureDataset(train, args.feature_dir)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_studies,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = create_feature_multiseries_model(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        dropout=0.1,
    ).to(device)
    projection = torch.nn.Linear(args.hidden_dim, report_embeddings.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projection.parameters()),
        lr=args.lr,
        weight_decay=0.05,
    )
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            study_embedding = model(
                batch["features"].to(device),
                batch["plane_ids"].to(device),
                batch["fluid"].to(device),
                batch["fat_sup"].to(device),
                batch["series_mask"].to(device),
                batch["slice_mask"].to(device),
                return_embeddings=True,
            )
            image_embedding = torch.nn.functional.normalize(projection(study_embedding), dim=-1)
            rows = torch.tensor([uid_to_row[uid] for uid in batch["study_uid"]])
            text_embedding = report_embeddings[rows].to(device)
            logits = image_embedding @ text_embedding.T / args.temperature
            target = torch.arange(len(logits), device=device)
            loss = (
                torch.nn.functional.cross_entropy(logits, target)
                + torch.nn.functional.cross_entropy(logits.T, target)
            ) / 2.0
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": epoch, "loss": mean_loss})
        print(f"epoch {epoch}: contrastive_loss={mean_loss:.5f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "projection": projection.state_dict(),
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "text_model": args.text_model,
            "seed": args.seed,
            "history": history,
        },
        args.out,
    )
    args.out.with_suffix(".json").write_text(json.dumps(history, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
