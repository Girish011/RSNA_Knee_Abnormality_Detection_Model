#!/usr/bin/env python3
"""Build v8 candidate labels = v6c base overlaid with v7∩Qwen consensus on TR/EL.

Requires:
  --base     adopted labels (e.g. weak_labels_v6c.csv)
  --v7       v7 extractor output (StudyInstanceUID + 12 labels [+ conf])
  --qwen     constrained-Qwen fills (same schema; typically TR/EL subset or full)
  --reports  optional train.csv with Report column to restrict overlay to TR/EL
             studies (recommended). Without it, consensus overlays everywhere
             both sources agree.

Writes a candidate CSV plus agreement stats. Does NOT train; gate + full-58 gold
OOF after a frozen-B 5-fold is the keep/kill ruler (compare to v6c 0.7023).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rsna_knee.constants import LABEL_COLS, SUBMISSION_ID_COL
from rsna_knee.text.label_consensus import agreement_stats, intersect_labels, overlay_consensus
from rsna_knee.text.weak_labels_v7 import detect_language


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[SUBMISSION_ID_COL] = df[SUBMISSION_ID_COL].astype(str)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--v7", type=Path, required=True)
    ap.add_argument("--qwen", type=Path, required=True)
    ap.add_argument("--reports", type=Path, default=None, help="train.csv with Report for lang filter")
    ap.add_argument("--out", type=Path, required=True, help="Output candidate CSV path")
    ap.add_argument(
        "--stats-out",
        type=Path,
        default=None,
        help="Optional agreement-stats CSV (defaults to <out>.agree.csv)",
    )
    args = ap.parse_args()

    base = _load(args.base)
    v7 = _load(args.v7)
    qwen = _load(args.qwen)

    consensus = intersect_labels(v7, qwen)
    stats = agreement_stats(v7, qwen)
    stats_path = args.stats_out or Path(str(args.out) + ".agree.csv")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(stats_path, index=False)
    both = int(stats["n_both_commit"].sum())
    agree = int(stats["n_agree"].sum())
    print(f"v7∩Qwen agreement: {agree}/{both} cells ({agree / both if both else float('nan'):.3f})")
    print(f"wrote {stats_path}")

    if args.reports is not None:
        reports = _load(args.reports)
        if "Report" not in reports.columns:
            raise SystemExit("--reports CSV needs a Report column")
        lang = reports[[SUBMISSION_ID_COL, "Report"]].copy()
        lang["lang"] = lang["Report"].map(lambda r: detect_language("" if pd.isna(r) else str(r)))
        trel_uids = set(lang.loc[lang["lang"].isin(["tr", "el"]), SUBMISSION_ID_COL])
        # Restrict consensus to TR/EL studies only.
        consensus = consensus[consensus[SUBMISSION_ID_COL].isin(trel_uids)].copy()
        print(f"overlay restricted to {len(consensus)} TR/EL studies")

    out = overlay_consensus(base, consensus, only_where_base_isna=False)
    # Ensure schema: id + labels (+ conf if present on base).
    keep = [SUBMISSION_ID_COL, *LABEL_COLS]
    for c in LABEL_COLS:
        conf = f"{c}__conf"
        if conf in out.columns:
            keep.append(conf)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out[keep].to_csv(args.out, index=False)
    known = int(out[LABEL_COLS].notna().sum().sum())
    print(f"wrote {args.out} ({len(out)} studies, {known} known cells)")


if __name__ == "__main__":
    main()
