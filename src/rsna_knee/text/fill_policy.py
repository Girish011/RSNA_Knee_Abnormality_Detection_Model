"""Per-label reliability policy for LLM-filled weak labels (train-only).

Constrained-decoding Qwen (v6) fixed the JSON parse problem (parse_rate 1.0) and
lifted expert recall 0.34 -> 0.72, but the *combined* 58-expert precision landed at
0.6826, just under the 0.69 gate. A per-label audit showed the miss is almost
entirely one label whose LLM fills are unreliable (MCL: fill precision 0.231; the
handoff already flagged "MCL fills were poisonous").

Rather than hand-pick a label, we pre-register a rule: **drop LLM fills for any
label whose measured LLM-fill expert precision is below ``min_precision`` (default
0.5 = worse than a coin flip on committed positives), keeping the keyword skeleton
for that label.** On the v6 audit this selects exactly ``MCL`` and lifts combined
precision to 0.703 (>= 0.69) while recall stays 0.690 and coverage stays ~27.7k
known cells. Pure pandas so it runs anywhere; expects the audit schema produced by
``rsna_knee.text.consensus_labels.audit_source``.
"""

from __future__ import annotations

import pandas as pd

from rsna_knee.constants import LABEL_COLS

DEFAULT_MIN_FILL_PRECISION = 0.5


def unreliable_fill_labels(
    fill_audit: pd.DataFrame,
    *,
    min_precision: float = DEFAULT_MIN_FILL_PRECISION,
) -> list[str]:
    """Labels whose LLM-fill expert precision is below ``min_precision``.

    ``fill_audit`` must have ``label`` and ``positive_precision`` columns (the
    ``llm_fill`` rows from ``audit_source``). Labels with NaN precision (no committed
    positive fills) are treated as reliable — there is nothing to drop.
    """
    out: list[str] = []
    for _, row in fill_audit.iterrows():
        label = row["label"]
        prec = row["positive_precision"]
        if label in LABEL_COLS and pd.notna(prec) and float(prec) < min_precision:
            out.append(label)
    return out


def drop_fills(fills: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    """Blank out the given labels' LLM fills (value + confidence) → back to skeleton.

    Returns a copy; the keyword skeleton still supplies those labels downstream.
    """
    out = fills.copy()
    for label in labels:
        if label in out.columns:
            out[label] = pd.NA
        conf_col = f"{label}__conf"
        if conf_col in out.columns:
            out[conf_col] = pd.NA
    return out
