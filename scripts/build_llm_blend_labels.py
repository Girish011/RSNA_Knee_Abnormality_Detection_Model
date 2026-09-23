#!/usr/bin/env python3
"""Build first-train soft-label files (DECISIONS 2026-09-24).

v1: mean of the pilkwang and dreaddevelopment LLM soft probabilities.
v2: same, but where pilk's verdict is UNK (it writes a flat 0.28; experts are positive in only
    ~14% of those cells) use dread alone.
Expert labels override the annotated studies in the CSV; the parquet (for the public trainer,
which validates on the experts itself) holds only non-expert studies. Train-only artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rsna_knee.constants import LABEL_COLS

PUB = Path("data/external/public_labels")
OUT = Path("data/processed")


def build(version: str) -> pd.DataFrame:
    train = pd.read_csv("data/raw/train.csv")
    uids = train["StudyInstanceUID"]
    pilk_raw = pd.read_csv(PUB / "pilk" / "report_labels_v2.csv").set_index("StudyInstanceUID").reindex(uids)
    dread = pd.read_csv(PUB / "dread" / "labels_llm_soft.csv").set_index("StudyInstanceUID").reindex(uids)[LABEL_COLS]
    pilk = pilk_raw[LABEL_COLS].copy()
    if version == "v2":
        for c in LABEL_COLS:
            unk = (pilk_raw[f"{c}__verdict"] == "UNK") & dread[c].notna()
            pilk.loc[unk, c] = float("nan")
    blend = pd.concat([pilk, dread]).groupby(level=0).mean().reindex(uids)

    expert = train.set_index("StudyInstanceUID")[LABEL_COLS]
    is_expert = expert.notna().all(axis=1)
    blend.loc[is_expert] = expert.loc[is_expert].astype(float)
    blend["source"] = "llm_blend"
    blend.loc[is_expert, "source"] = "expert"
    blend.loc[~is_expert & dread[LABEL_COLS[0]].isna(), "source"] = "pilk_only"
    assert blend[LABEL_COLS].notna().all().all(), "study with no label source"
    return blend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v2", choices=["v1", "v2"])
    a = ap.parse_args()
    blend = build(a.version)
    OUT.mkdir(parents=True, exist_ok=True)
    csv = OUT / f"labels_llm_blend_{a.version}.csv"
    blend.reset_index().to_csv(csv, index=False)
    nonexp = blend[blend["source"] != "expert"]
    nonexp[LABEL_COLS].reset_index().to_parquet(csv.with_suffix(".parquet"), index=False)
    summary = {"n": len(blend), "source_counts": blend["source"].value_counts().to_dict(),
               "mean_soft": nonexp[LABEL_COLS].mean().round(3).to_dict()}
    print(json.dumps(summary, indent=2))
    print(f"wrote {csv} and {csv.with_suffix('.parquet')}")


if __name__ == "__main__":
    main()
