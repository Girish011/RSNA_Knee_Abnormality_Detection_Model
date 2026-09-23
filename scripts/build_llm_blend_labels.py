#!/usr/bin/env python3
"""Build the first-train soft-label file (DECISIONS 2026-09-24).

Mean of the pilkwang and dreaddevelopment LLM soft probabilities (either alone where the
other is missing), expert labels override the annotated studies. Train-only artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rsna_knee.constants import LABEL_COLS

PUB = Path("data/external/public_labels")
OUT = Path("data/processed/labels_llm_blend_v1.csv")


def main() -> None:
    train = pd.read_csv("data/raw/train.csv")
    ids = train[["StudyInstanceUID"]]
    pilk = ids.merge(pd.read_csv(PUB / "pilk" / "report_labels_v2.csv")[["StudyInstanceUID", *LABEL_COLS]],
                     on="StudyInstanceUID", how="left").set_index("StudyInstanceUID")
    dread = ids.merge(pd.read_csv(PUB / "dread" / "labels_llm_soft.csv")[["StudyInstanceUID", *LABEL_COLS]],
                      on="StudyInstanceUID", how="left").set_index("StudyInstanceUID")
    blend = pd.concat([pilk, dread]).groupby(level=0).mean().loc[ids["StudyInstanceUID"]]

    expert = train.set_index("StudyInstanceUID")[LABEL_COLS]
    is_expert = expert.notna().all(axis=1)
    blend.loc[is_expert] = expert.loc[is_expert].astype(float)
    blend["source"] = "llm_blend"
    blend.loc[is_expert, "source"] = "expert"
    blend.loc[~is_expert & dread[LABEL_COLS[0]].isna(), "source"] = "pilk_only"

    assert blend[LABEL_COLS].notna().all().all(), "study with no label source"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    blend.reset_index().to_csv(OUT, index=False)
    summary = {"n": len(blend), "source_counts": blend["source"].value_counts().to_dict(),
               "mean_soft": blend.loc[~is_expert, LABEL_COLS].mean().round(3).to_dict()}
    print(json.dumps(summary, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
