#!/usr/bin/env python3
"""Paired, fixed-EMA pilot diagnostics. Report agreement is not expert test AUC.

Both arms are scored against the SAME report-control targets/mask. Optional expert58
scores are explicitly adaptively reused development diagnostics. No weight fitting,
model selection, cloud operations or submissions occur here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from rsna_knee.constants import LABEL_COLS  # noqa: E402
from rsna_knee.definition_targets import dump_json, sha256, validate_arrays  # noqa: E402

MATCH_FIELDS = ("seed", "fold", "bs", "accum", "epochs", "lr", "head_lr", "wd", "ema",
                "labels_sha256", "train_csv_sha256", "series_csv_sha256", "corpus_geometry_sha256",
                "code_sha256", "geometry_helper_sha256", "init_sha256", "init_source")


def validate_receipts(control: dict, candidate: dict, *, allow_aux_change: bool = False) -> dict:
    for name, rec in (("control", control), ("candidate", candidate)):
        missing = [k for k in (*MATCH_FIELDS, "protocol", "target_mode", "aux_weight") if k not in rec]
        if missing:
            raise ValueError(f"{name} receipt missing required matching fields: {missing}")
        if rec["protocol"].get("checkpoint_policy") != "fixed_ema":
            raise ValueError(f"{name} must use fixed_ema, never best expert epoch")
        if rec.get("smoke"):
            raise ValueError("smoke outputs cannot be treated as a full pilot")
    if control["target_mode"] != "control" or candidate["target_mode"] != "definition":
        raise ValueError("expected control vs definition target modes")
    differences = [k for k in (*MATCH_FIELDS, "protocol") if control[k] != candidate[k]]
    if differences:
        raise ValueError(f"unmatched experimental settings: {differences}")
    aux_diff = control["aux_weight"] != candidate["aux_weight"]
    if aux_diff and not allow_aux_change:
        raise ValueError("aux_weight changed: use --allow-aux-change to declare the combined target+auxiliary hypothesis")
    return {"matched_fields": list(MATCH_FIELDS) + ["protocol"],
            "intentional_changes": ["target_mode"] + (["aux_weight"] if aux_diff else []),
            "interpretation": "combined definition targets plus severity auxiliary" if aux_diff else "definition targets only",
            "checkpoint_policy": "fixed_ema"}


def load_predictions(path: Path, receipt: dict, checkpoint_sha256: str | None = None,
                     *, require_metadata: bool = True) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        for key in ("ids", "pred", "labels", "protocol"):
            if key not in z:
                raise ValueError(f"{path}: missing {key}")
        ids, pred = z["ids"].astype(str), z["pred"].astype(float)
        if list(z["labels"].astype(str)) != LABEL_COLS:
            raise ValueError(f"{path}: unexpected label order")
        if json.loads(str(z["protocol"].item())) != receipt["protocol"]:
            raise ValueError(f"{path}: prediction protocol differs from receipt")
        if require_metadata and "metadata" not in z:
            raise ValueError(f"{path}: missing training contract metadata")
        if "metadata" in z and json.loads(str(z["metadata"].item())) != receipt:
            raise ValueError(f"{path}: prediction metadata differs from training contract")
        if checkpoint_sha256 is not None and ("checkpoint_sha256" not in z or str(z["checkpoint_sha256"].item()) != checkpoint_sha256):
            raise ValueError(f"{path}: checkpoint SHA differs from completed training receipt")
    if len(set(ids)) != len(ids) or pred.shape != (len(ids), len(LABEL_COLS)):
        raise ValueError(f"{path}: duplicate IDs or malformed prediction shape")
    if not np.isfinite(pred).all() or np.any((pred < 0) | (pred > 1)):
        raise ValueError(f"{path}: predictions must be finite probabilities")
    return ids, pred


def align_pairs(control_ids: np.ndarray, control: np.ndarray, candidate_ids: np.ndarray,
                candidate: np.ndarray, expected_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expected = set(map(str, expected_ids))
    if set(control_ids) != expected or set(candidate_ids) != expected:
        raise ValueError("prediction IDs must exactly cover the same complete prescribed evaluation set")
    ca, cb = {u: i for i, u in enumerate(control_ids)}, {u: i for i, u in enumerate(candidate_ids)}
    return control[[ca[u] for u in expected_ids]], candidate[[cb[u] for u in expected_ids]]


def class_auc(y: np.ndarray, score: np.ndarray) -> float:
    p, n = int(y.sum()), len(y)
    if p == 0 or p == n:
        return float("nan")
    ranks = rankdata(score)
    return float((np.sum(ranks[y.astype(bool)]) - p * (p + 1) / 2) / (p * (n - p)))


def paired_score(y: np.ndarray, mask: np.ndarray, control: np.ndarray, candidate: np.ndarray,
                 *, n_bootstrap: int = 2000, seed: int = 42,
                 groups: np.ndarray | None = None) -> dict:
    """Study-paired bootstrap, or cluster-paired with explicitly verified groups.

    The estimand is the same fixed set of scorable findings in every replicate.
    Draws missing either class in any such finding are rejected, and counted.
    The interval does not account for historical adaptive selection.
    """
    if y.shape != mask.shape or y.shape != control.shape or y.shape != candidate.shape:
        raise ValueError("paired arrays must have identical shapes")
    if y.shape[1] != len(LABEL_COLS) or n_bootstrap < 1:
        raise ValueError("expected twelve findings and positive bootstrap count")
    y = y >= .5
    mask = mask.astype(bool)
    rows, eligible = [], []
    for j, label in enumerate(LABEL_COLS):
        m = mask[:, j]
        positive, negative = int(y[m, j].sum()), int((~y[m, j]).sum())
        ac, ad = class_auc(y[m, j], control[m, j]), class_auc(y[m, j], candidate[m, j])
        row = {"label": label, "positive": positive, "negative": negative, "masked": int((~m).sum()),
               "control_auc": ac if np.isfinite(ac) else None,
               "candidate_auc": ad if np.isfinite(ad) else None,
               "delta": ad - ac if np.isfinite(ac) and np.isfinite(ad) else None,
               "small_class_warning": min(positive, negative) < 20}
        rows.append(row)
        if row["delta"] is not None:
            eligible.append(j)
    if not eligible:
        raise ValueError("no finding has both classes among masked-known cells")
    delta = float(np.mean([rows[j]["delta"] for j in eligible]))
    group_array = np.arange(len(y)) if groups is None else np.asarray(groups)
    if group_array.shape != (len(y),):
        raise ValueError("one verified group ID required per study")
    unique = np.unique(group_array)
    members = [np.flatnonzero(group_array == g) for g in unique]
    rng = np.random.default_rng(seed)
    draws, attempts = [], 0
    while len(draws) < n_bootstrap and attempts < n_bootstrap * 100:
        attempts += 1
        ix = np.concatenate([members[j] for j in rng.integers(len(members), size=len(members))])
        ds = []
        for j in eligible:
            take = ix[mask[ix, j]]
            a = class_auc(y[take, j], control[take, j])
            b = class_auc(y[take, j], candidate[take, j])
            if not np.isfinite(a) or not np.isfinite(b):
                break
            ds.append(b - a)
        if len(ds) == len(eligible):
            draws.append(float(np.mean(ds)))
    if len(draws) != n_bootstrap:
        raise ValueError("insufficient valid bootstrap draws; class counts too small for this paired estimate")
    return {"n_studies": len(y), "n_resampling_units": len(unique),
            "resampling_unit": "study" if groups is None else "verified_patient_cluster",
            "scorable_findings": [LABEL_COLS[j] for j in eligible],
            "control_macro_auc": float(np.mean([rows[j]["control_auc"] for j in eligible])),
            "candidate_macro_auc": float(np.mean([rows[j]["candidate_auc"] for j in eligible])),
            "macro_delta": delta, "paired_delta_ci95": np.quantile(draws, [.025, .975]).tolist(),
            "paired_delta_bootstrap_sd": float(np.std(draws)), "bootstrap_seed": seed,
            "bootstrap_accepted": len(draws), "bootstrap_rejected": attempts - len(draws),
            "bootstrap_policy": "same studies/clusters for both arms; reject draws missing a class in any fixed scorable finding",
            "per_finding": rows,
            "limitations": ["confidence interval is conditional on the existing cases and fixed recipe; not a correction for adaptive reuse",
                            "study-ID disjointness does not establish clinical patient independence",
                            "small class counts are descriptive only; do not tune per-finding ensemble weights on this output"]}


def _groups(path: Path | None, ids: np.ndarray) -> np.ndarray | None:
    if path is None:
        return None
    data = json.loads(path.read_text())
    if data.get("verified_patient_ids") is not True or not data.get("provenance"):
        raise ValueError("patient mapping needs verified_patient_ids=true and a provenance description")
    if not set(ids) <= set(data.get("groups", {})):
        raise ValueError("patient mapping does not cover evaluated studies")
    groups = np.array([str(data["groups"][u]) for u in ids])
    if np.any(np.isin(groups, ["", "None", "nan"])):
        raise ValueError("patient groups cannot be missing")
    return groups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--control-receipt", type=Path, required=True)
    ap.add_argument("--candidate-receipt", type=Path, required=True)
    ap.add_argument("--labels", type=Path, default=REPO / "data/processed/definition_v1.npz")
    ap.add_argument("--train-csv", type=Path, default=REPO / "data/raw/train.csv")
    ap.add_argument("--control-expert", type=Path)
    ap.add_argument("--candidate-expert", type=Path)
    ap.add_argument("--patient-groups", type=Path)
    ap.add_argument("--allow-aux-change", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cr, dr = json.loads(a.control_receipt.read_text()), json.loads(a.candidate_receipt.read_text())
    for name, receipt in (("control", cr), ("candidate", dr)):
        if receipt.get("status") != "complete" or "contract" not in receipt or not receipt.get("checkpoint_sha256"):
            raise ValueError(f"{name}: a completed trainer receipt with contract and checkpoint SHA is required")
    rc, rd = cr["contract"], dr["contract"]
    matched = validate_receipts(rc, rd, allow_aux_change=a.allow_aux_change)
    if sha256(a.labels) != rc["labels_sha256"] or sha256(a.train_csv) != rc["train_csv_sha256"]:
        raise ValueError("evaluation label/train CSV hashes do not match training receipts")
    with np.load(a.labels, allow_pickle=False) as z:
        labels = {k: z[k] for k in z.files}
    validate_arrays(labels)
    if bool(labels.get("is_expert_diagnostic", False)):
        raise ValueError("expert diagnostic file cannot supply training targets")
    fold = int(rc["fold"])
    held = labels["folds"] == fold
    ids = labels["ids"][held].astype(str)
    if fold not in range(5) or len(ids) == 0:
        raise ValueError("receipt fold must identify a complete frozen nonexpert fold 0..4")
    cids, cp = load_predictions(a.control, rc, cr["checkpoint_sha256"])
    dids, dp = load_predictions(a.candidate, rd, dr["checkpoint_sha256"])
    cp, dp = align_pairs(cids, cp, dids, dp, ids)
    result = {"experiment": matched, "report_diagnostic": paired_score(
        labels["control_target"][held], labels["control_mask"][held], cp, dp,
        n_bootstrap=a.bootstrap, seed=a.seed, groups=_groups(a.patient_groups, ids)),
        "report_metric_interpretation": "agreement with the SAME thresholded raw report consensus, not expert competition truth",
        "provenance": {str(p): sha256(p) for p in [a.control, a.candidate, a.control_receipt, a.candidate_receipt, a.labels, a.train_csv]},
        "decision": "descriptive paired diagnostics only; no automatic keep/kill, weight tuning, or leaderboard prediction"}
    if bool(a.control_expert) != bool(a.candidate_expert):
        raise ValueError("both expert prediction files must be supplied together")
    if a.control_expert:
        tr = pd.read_csv(a.train_csv, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
        gold = tr.dropna(subset=LABEL_COLS)[LABEL_COLS]
        gi = gold.index.to_numpy(str)
        ci, ce = load_predictions(a.control_expert, rc, cr["checkpoint_sha256"], require_metadata=False)
        di, de = load_predictions(a.candidate_expert, rd, dr["checkpoint_sha256"], require_metadata=False)
        ce, de = align_pairs(ci, ce, di, de, gi)
        result["expert58_reused_development_diagnostic"] = paired_score(
            gold.to_numpy(), np.ones(gold.shape, bool), ce, de,
            n_bootstrap=a.bootstrap, seed=a.seed, groups=_groups(a.patient_groups, gi))
        result["expert58_warning"] = "adaptively reused development cases, NOT an untouched holdout or unbiased test estimate; not used to fit this score"
        result["provenance"].update({str(p): sha256(p) for p in [a.control_expert, a.candidate_expert]})
    if a.patient_groups:
        result["provenance"][str(a.patient_groups)] = sha256(a.patient_groups)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    dump_json(a.out, result)
    score = result["report_diagnostic"]
    print(f"report agreement: control {score['control_macro_auc']:.6f}, candidate {score['candidate_macro_auc']:.6f}, "
          f"delta {score['macro_delta']:+.6f}, paired 95% CI {score['paired_delta_ci95']}")
    print(f"wrote {a.out}; fixed-EMA study-paired diagnostic, not a leaderboard forecast")


if __name__ == "__main__":
    main()
