#!/usr/bin/env python3
"""Freeze study-level folds from local train.csv metadata (no DICOMs needed)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rsna_knee.data.folds import make_study_folds, save_folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv",
        default="data/raw/train.csv",
        type=Path,
    )
    parser.add_argument(
        "--out",
        default="data/folds/folds_v1.csv",
        type=Path,
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-col",
        default=None,
        help="Optional site/scanner proxy column if present in train.csv",
    )
    args = parser.parse_args()

    if not args.train_csv.exists():
        raise SystemExit(
            f"Missing {args.train_csv}. Run ./scripts/download_metadata.sh first."
        )

    train = pd.read_csv(args.train_csv)
    folds = make_study_folds(
        train,
        n_folds=args.n_folds,
        seed=args.seed,
        group_col=args.group_col,
    )
    path = save_folds(folds, args.out)
    print(f"Wrote {path} ({len(folds)} studies, {args.n_folds} folds)")
    print(folds["fold"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
