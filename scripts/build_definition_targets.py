#!/usr/bin/env python3
"""Build frozen report-only definition targets, matched raw control and provenance.

    .venv/bin/python scripts/build_definition_targets.py

The default writes data/processed/definition_v1.npz, .manifest.json, .targets.parquet,
and .summary.csv. Expert58 diagnostics are opt-in and always a separate artifact.
No raw report text is exported. This script never imports an LLM client.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from rsna_knee.constants import LABEL_COLS  # noqa: E402
from rsna_knee.definition_targets import (  # noqa: E402
    build_arrays, coverage, dump_json, policy_manifest, sha256,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, default=REPO / "data/raw/train.csv")
    ap.add_argument("--grades", type=Path, default=REPO / "data/processed/labels_graded_v1.parquet")
    ap.add_argument("--pilk", type=Path, default=REPO / "data/external/public_labels/pilk/report_labels_v2.csv")
    ap.add_argument("--dread", type=Path, default=REPO / "data/external/public_labels/dread/labels_llm_soft.csv")
    ap.add_argument("--folds", type=Path, default=REPO / "agent/protocol/folds_nonexpert_5.json")
    ap.add_argument("--out", type=Path, default=REPO / "data/processed/definition_v1.npz")
    ap.add_argument("--expected-n", type=int, default=4349)
    ap.add_argument("--expert-diagnostic", action="store_true", help="write a separate report-only expert58 diagnostic NPZ/JSON; never alters targets")
    args = ap.parse_args()
    if args.out.suffix != ".npz":
        raise SystemExit("--out must have .npz extension")
    train = pd.read_csv(args.train, dtype={"StudyInstanceUID": str})
    grades = pd.read_parquet(args.grades)
    pilk = pd.read_csv(args.pilk, dtype={"StudyInstanceUID": str})
    dread = pd.read_csv(args.dread, dtype={"StudyInstanceUID": str})
    fold_spec = json.loads(args.folds.read_text())
    arrays = build_arrays(train, grades, pilk, dread, fold_spec["folds"])
    if len(arrays["ids"]) != args.expected_n:
        raise SystemExit(f"expected {args.expected_n} nonexperts, found {len(arrays['ids'])}")
    counts = coverage(arrays)
    # Refuse a target set that would train an all-negative finding head.
    if any(x["positive"] == 0 or x["negative"] == 0 for x in counts):
        raise SystemExit("at least one definition finding lacks positive or negative supervision")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    summary_path = args.out.with_suffix(".summary.csv")
    pd.DataFrame(counts).to_csv(summary_path, index=False)
    n, k = arrays["target"].shape
    frame = {"StudyInstanceUID": np.repeat(arrays["ids"], k), "Finding": np.tile(arrays["labels"], n),
             "fold": np.repeat(arrays["folds"], k)}
    for name in ("grade", "target", "mask", "confidence", "strong_mask", "weak_mask", "state", "reason",
                 "control_target", "control_mask", "control_confidence", "control_state", "control_reason"):
        frame[name] = arrays[name].reshape(-1)
    for j, threshold in enumerate((2, 3, 4)):
        for name in ("ordinal_target", "ordinal_mask", "ordinal_confidence"):
            frame[f"{name}_ge{threshold}"] = arrays[name][..., j].reshape(-1)
    table_path = args.out.with_suffix(".targets.parquet")
    pd.DataFrame(frame).to_parquet(table_path, index=False)
    sources = []
    for role in ("train", "grades", "pilk", "dread", "folds"):
        path = getattr(args, role)
        sources.append({"role": role, "path": str(path.resolve()), "sha256": sha256(path)})
    for role, path in [("graded_extractor_definition", REPO / "scripts/graded_labels.py"),
                       ("pilk_extractor_definition", REPO / "data/external/public_labels/pilk/api_labeler.py")]:
        if path.exists():
            sources.append({"role": role, "path": str(path), "sha256": sha256(path)})
    manifest = policy_manifest()
    manifest.update({"artifact": {"path": str(args.out.resolve()), "sha256": sha256(args.out)},
                     "sources": sources, "labels": LABEL_COLS, "n_studies": n,
                     "excluded_expert_count": int(train[LABEL_COLS].notna().all(axis=1).sum()),
                     "fold_counts": dict(Counter(map(int, arrays["folds"]))),
                     "fold_seed": fold_spec.get("seed"), "per_label_counts": counts,
                     "code_sha256": {str(Path(__file__).resolve()): sha256(__file__),
                                     str(REPO / "src/rsna_knee/definition_targets.py"): sha256(REPO / "src/rsna_knee/definition_targets.py")},
                     "additional_artifacts": [{"path": str(p.resolve()), "sha256": sha256(p)} for p in [summary_path, table_path]],
                     "source_lineage": "existing competition TRAIN report readers only; no labels_r*, labels_t2, fused labels or image predictions",
                     "training_interface": "definition: target, mask, confidence; control: control_target, control_mask, control_confidence; per-cell weight=mask*confidence; ordinal auxiliary is separate",
                     "raw_report_text_exported": False})
    dump_json(args.out.with_suffix(".manifest.json"), manifest)
    if args.expert_diagnostic:
        ex = build_arrays(train, grades, pilk, dread, fold_spec["folds"], expert_diagnostic=True)
        ex_path = args.out.with_suffix(".expert_diagnostic.npz")
        np.savez_compressed(ex_path, **ex)
        gold = train.set_index("StudyInstanceUID").loc[ex["ids"], LABEL_COLS].to_numpy()
        diagnostics = []
        for j, label in enumerate(LABEL_COLS):
            m, pred = ex["mask"][:, j], ex["target"][:, j] >= .5
            y = gold[:, j] >= .5
            diagnostics.append({"label": label, "supervised_cells": int(m.sum()),
                                "weak_cells": int(ex["weak_mask"][:, j].sum()),
                                "tp": int((m & pred & y).sum()), "fp": int((m & pred & ~y).sum()),
                                "tn": int((m & ~pred & ~y).sum()), "fn": int((m & ~pred & y).sum())})
        dump_json(args.out.with_suffix(".expert_diagnostic.json"),
                  {"warning": "reused expert58 descriptive diagnostic ONLY; no fitting or model-selection gate; missing cells excluded",
                   "artifact_sha256": sha256(ex_path), "counts": diagnostics})
    print(pd.DataFrame(counts)[["label", "positive", "negative", "masked", "strong_positive", "weak_positive", "ordinal_known"]].to_string(index=False))
    print(f"wrote {args.out} | {n} nonexperts | sha256 {manifest['artifact']['sha256']}")


if __name__ == "__main__":
    main()
