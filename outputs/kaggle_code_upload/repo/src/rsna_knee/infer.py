"""Inference helpers and CLI for writing submission.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rsna_knee.constants import LABEL_COLS, SUBMISSION_ID_COL


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
