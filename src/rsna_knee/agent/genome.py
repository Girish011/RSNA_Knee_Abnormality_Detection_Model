"""Typed recipe genome: the only knobs the planner may turn.

The planner never writes code. It proposes a genome, which is validated against the menu
below and compiled into arguments for the vendored public trainer (``runpod/train_knee.py``).
Protocol fields (fold, holdout, seed) are fixed so every online node is paired with its
parent (same split, same seed, same corpus unless the corpus is the change).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LABEL_REGISTRY = REPO / "agent" / "labels" / "registry.json"

ARCHS = {
    "coatnet384": ("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", 384),
    "convnext_b384": ("convnext_base.fb_in22k_ft_in1k_384", 384),
}
CORPORA = {"raptor44": True, "dense80": False}  # value: available to the loop
MENU = {
    "corpus": list(CORPORA),
    "arch": list(ARCHS),
    "epochs": [12, 16, 20],
    "k": [8, 12, 16],
    "k_eval": [24, 42],
    "bb_lr": [1e-5, 2e-5, 3e-5, 5e-5],
    "head_lr": [5e-4, 1e-3, 2e-3],
    "wd": [0.02, 0.05],
    "freeze_blocks": [0, 1, 2],
}
PROTOCOL = {"seed": 42, "folds": 5, "fold": 0}
BASE = {"corpus": "raptor44", "labels": "dread", "arch": "coatnet384", "epochs": 16, "k": 12,
        "k_eval": 24, "bb_lr": 3e-5, "head_lr": 1e-3, "wd": 0.02, "freeze_blocks": 0}

# Rough 4090 hours for a 4/5-fold run of each arch at 16 epochs (3090 is ~1.6x).
HOURS_16EP = {"coatnet384": 2.6, "convnext_b384": 2.2}


def label_registry() -> dict[str, dict]:
    """name -> {"file": parquet name under data/processed, "audit": {...}}; dread/blend_v2 built in."""
    reg = {"dread": {"file": "labels_dread.parquet"},
           "blend_v2": {"file": "labels_llm_blend_v2.parquet"}}
    if LABEL_REGISTRY.exists():
        reg.update(json.loads(LABEL_REGISTRY.read_text()))
    return reg


def validate(genome: dict) -> list[str]:
    errs = []
    unknown = set(genome) - set(MENU) - {"labels"}
    if unknown:
        errs.append(f"unknown fields {sorted(unknown)}")
    for k, allowed in MENU.items():
        if k not in genome:
            errs.append(f"missing {k}")
        elif genome[k] not in allowed:
            errs.append(f"{k}={genome[k]!r} not in {allowed}")
    if genome.get("corpus") in CORPORA and not CORPORA[genome["corpus"]]:
        errs.append(f"corpus {genome['corpus']} not available yet")
    if genome.get("labels") not in label_registry():
        errs.append(f"labels {genome.get('labels')!r} not in registry {sorted(label_registry())}")
    return errs


def diff(parent: dict, child: dict) -> dict:
    return {k: child[k] for k in child if parent.get(k) != child[k]}


def to_cli(genome: dict, bundle_dir: str = ".") -> tuple[str, str]:
    """(labels parquet file name, trainer CLI args) for one arm."""
    arch, res = ARCHS[genome["arch"]]
    args = (f"--arch {arch} --res {res} --epochs {genome['epochs']} --bs 8 --k {genome['k']} "
            f"--k_eval {genome['k_eval']} --bb_lr {genome['bb_lr']} --head_lr {genome['head_lr']} "
            f"--wd {genome['wd']} --freeze_blocks {genome['freeze_blocks']} --grad_ckpt --norm imagenet "
            f"--seed {PROTOCOL['seed']} --folds {PROTOCOL['folds']} --fold {PROTOCOL['fold']} "
            f"--fold_file {bundle_dir}/folds_nonexpert_5.json --holdout {bundle_dir}/expert_holdout.json "
            f"--workers 8")
    return label_registry()[genome["labels"]]["file"], args


def est_hours(genome: dict, gpu_factor: float = 1.6) -> float:
    return HOURS_16EP[genome["arch"]] * genome["epochs"] / 16 * gpu_factor + 0.5
