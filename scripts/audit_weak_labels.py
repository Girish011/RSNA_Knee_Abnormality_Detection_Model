#!/usr/bin/env python3
"""Audit weak-label extractor vs expert-labeled subset; export weak_labels_v2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from rsna_knee.constants import LABEL_COLS
from rsna_knee.text.weak_labels import apply_weak_labels, weak_labels_for_report


def main() -> None:
    train = pd.read_csv("data/raw/train.csv")
    expert = train.loc[train[LABEL_COLS].notna().all(axis=1)].copy()
    pure = pd.DataFrame([weak_labels_for_report(str(r)) for r in expert["Report"]])

    rows = []
    for c in LABEL_COLS:
        y = expert[c].astype(int).to_numpy()
        pred = pure[c].astype(float).to_numpy()
        conf = pure[f"{c}__conf"].astype(float).to_numpy()
        m = conf >= 0.5
        rows.append(
            {
                "label": c,
                "expert_pos": int(y.sum()),
                "pred_pos": int((pred == 1).sum()),
                "acc": float(accuracy_score(y, pred)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "prec": float(precision_score(y, pred, zero_division=0)),
                "rec": float(recall_score(y, pred, zero_division=0)),
                "n_ge_0.5": int(m.sum()),
                "acc_ge_0.5": float(accuracy_score(y[m], pred[m])) if m.any() else np.nan,
                "f1_ge_0.5": float(f1_score(y[m], pred[m], zero_division=0)) if m.any() else np.nan,
            }
        )
    audit = pd.DataFrame(rows)
    out = Path("docs/audit")
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / "weak_label_v2_vs_expert.csv", index=False)
    print(audit.to_string(index=False))
    print(f"macro f1={audit['f1'].mean():.3f} prec={audit['prec'].mean():.3f} rec={audit['rec'].mean():.3f}")

    full = apply_weak_labels(train, min_confidence=0.5, expert_override=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    keep = ["StudyInstanceUID", "Report"] + LABEL_COLS + [f"{c}__conf" for c in LABEL_COLS]
    full[keep].to_csv("data/processed/weak_labels_v2.csv", index=False)
    print("wrote data/processed/weak_labels_v2.csv")

    # Coverage: how many studies get ≥1 supervised label
    mask_cols = LABEL_COLS
    n_any = int((full[mask_cols].notna().any(axis=1)).sum())
    n_fr_proxy = int(
        full["Report"]
        .astype(str)
        .str.contains(r"\b(?:le|la|les|des|une|du|au)\b", case=False, regex=True)
        .sum()
    )
    print(f"studies with any label: {n_any}/{len(full)}; rough FR-like reports: {n_fr_proxy}")


if __name__ == "__main__":
    main()
