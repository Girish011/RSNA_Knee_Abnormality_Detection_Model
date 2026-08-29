"""Intersect independent weak-label sources (train-only).

The v7 multilingual keyword extractor and constrained-Qwen fills agree on ~88% of
Turkish cells and ~77% of Greek cells where both commit. Keeping only the
agreement cells (v7 ∩ Qwen) yields a high-precision supervision signal for those
languages without trusting either method alone. Disagreements and one-sided
commits become abstentions (NaN).

This is the building block for **v8** labels: start from an adopted recipe
(e.g. v6c skeleton + reliable fills) and replace TR/EL cells with the consensus
where both methods agree. Never used at inference.
"""

from __future__ import annotations

import pandas as pd

from rsna_knee.constants import LABEL_COLS, SUBMISSION_ID_COL


def intersect_labels(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    labels: list[str] | None = None,
    id_col: str = SUBMISSION_ID_COL,
    how: str = "outer",
) -> pd.DataFrame:
    """Keep a label only where ``left`` and ``right`` both commit and agree.

    Both frames must contain ``id_col`` and the label columns. Optional
    ``<label>__conf`` columns are preserved as ``min(left_conf, right_conf)`` on
    agreement cells and NaN otherwise.

    Parameters
    ----------
    how:
        Study join mode. ``outer`` keeps every study from either source (default);
        ``inner`` keeps only studies present in both.
    """
    labs = list(labels) if labels is not None else list(LABEL_COLS)
    if id_col not in left.columns or id_col not in right.columns:
        raise ValueError(f"both frames need {id_col!r}")

    li = left.set_index(id_col)
    ri = right.set_index(id_col)
    if how == "inner":
        uids = li.index.intersection(ri.index)
    elif how == "outer":
        uids = li.index.union(ri.index)
    else:
        raise ValueError(f"how must be 'outer' or 'inner', got {how!r}")

    # Preserve first-seen order: left then new-from-right.
    ordered = list(dict.fromkeys([*li.index.tolist(), *ri.index.tolist()]))
    ordered = [u for u in ordered if u in set(uids)]

    rows: list[dict[str, object]] = []
    for uid in ordered:
        row: dict[str, object] = {id_col: uid}
        l_ok = uid in li.index
        r_ok = uid in ri.index
        lr = li.loc[uid] if l_ok else None
        rr = ri.loc[uid] if r_ok else None
        # Duplicate UIDs would yield a DataFrame; take first row.
        if isinstance(lr, pd.DataFrame):
            lr = lr.iloc[0]
        if isinstance(rr, pd.DataFrame):
            rr = rr.iloc[0]
        for lab in labs:
            lv = lr[lab] if lr is not None and lab in lr.index else pd.NA
            rv = rr[lab] if rr is not None and lab in rr.index else pd.NA
            conf_col = f"{lab}__conf"
            lc = (
                lr[conf_col]
                if lr is not None and conf_col in lr.index
                else pd.NA
            )
            rc = (
                rr[conf_col]
                if rr is not None and conf_col in rr.index
                else pd.NA
            )
            if pd.notna(lv) and pd.notna(rv) and float(lv) == float(rv):
                row[lab] = float(lv)
                if pd.notna(lc) and pd.notna(rc):
                    row[conf_col] = float(min(float(lc), float(rc)))
                elif pd.notna(lc):
                    row[conf_col] = float(lc)
                elif pd.notna(rc):
                    row[conf_col] = float(rc)
                else:
                    row[conf_col] = pd.NA
            else:
                row[lab] = pd.NA
                row[conf_col] = pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def agreement_stats(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    labels: list[str] | None = None,
    id_col: str = SUBMISSION_ID_COL,
) -> pd.DataFrame:
    """Per-label commit/agree rates where both sources have a non-null value."""
    labs = list(labels) if labels is not None else list(LABEL_COLS)
    merged = left[[id_col, *labs]].merge(
        right[[id_col, *labs]],
        on=id_col,
        how="inner",
        suffixes=("_l", "_r"),
    )
    rows = []
    for lab in labs:
        a = merged[f"{lab}_l"]
        b = merged[f"{lab}_r"]
        both = a.notna() & b.notna()
        n_both = int(both.sum())
        n_agree = int((both & (a.astype(float) == b.astype(float))).sum()) if n_both else 0
        rows.append(
            {
                "label": lab,
                "n_studies": len(merged),
                "n_both_commit": n_both,
                "n_agree": n_agree,
                "agree_rate": (n_agree / n_both) if n_both else float("nan"),
                "n_left_only": int((a.notna() & b.isna()).sum()),
                "n_right_only": int((a.isna() & b.notna()).sum()),
            }
        )
    return pd.DataFrame(rows)


def overlay_consensus(
    base: pd.DataFrame,
    consensus: pd.DataFrame,
    *,
    labels: list[str] | None = None,
    id_col: str = SUBMISSION_ID_COL,
    only_where_base_isna: bool = False,
) -> pd.DataFrame:
    """Write consensus cells onto ``base``.

    When ``only_where_base_isna`` is True, consensus fills gaps only (safe additive
    overlay). When False, consensus overwrites base wherever consensus commits —
    the intended v8 pattern for TR/EL high-precision replacement.
    """
    labs = list(labels) if labels is not None else list(LABEL_COLS)
    out = base.copy()
    for lab in labs:
        if lab not in out.columns:
            out[lab] = pd.NA
        conf_col = f"{lab}__conf"
        if conf_col not in out.columns:
            out[conf_col] = pd.NA

    ci = consensus.set_index(id_col)
    for i, row in out.iterrows():
        uid = str(row[id_col])
        if uid not in ci.index:
            continue
        cr = ci.loc[uid]
        if isinstance(cr, pd.DataFrame):
            cr = cr.iloc[0]
        for lab in labs:
            if lab not in cr.index or pd.isna(cr[lab]):
                continue
            if only_where_base_isna and pd.notna(out.at[i, lab]):
                continue
            out.at[i, lab] = float(cr[lab])
            conf_col = f"{lab}__conf"
            if conf_col in cr.index and pd.notna(cr[conf_col]):
                out.at[i, conf_col] = float(cr[conf_col])
            else:
                out.at[i, conf_col] = 1.0
    return out
