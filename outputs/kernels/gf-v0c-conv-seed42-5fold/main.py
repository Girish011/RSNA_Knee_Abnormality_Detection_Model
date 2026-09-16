"""gf_v0c stage 1: does the v0 recipe just need to finish converging?

Trains gf_v0 for 10 epochs instead of 5 (backbone frozen throughout, no LR schedule,
no unfreeze) and saves val predictions after EVERY epoch. That single run answers three
questions offline, with no extra GPU:

1. Is 5 epochs undertrained?  -> per-epoch true OOF gold-58 curve. Because there is no LR
   schedule and no unfreeze, epochs 0-4 here retrace the 5-epoch seed-42 run exactly, so
   "epoch 4 vs epoch 9" is a PAIRED comparison with zero seed noise. The kernel prints the
   epochs 0-4 weak-val against the recorded seed-42 values as a reproduction check.
2. Does checkpoint selection matter?  -> compares best-on-weak-val (the old policy, which
   picks against a ruler that disagrees with gold), final epoch, and last-k averaging.
3. Which policy should later A/Bs use?  -> whichever wins here becomes the fixed policy.

Ruler (DECISIONS 2026-09-16): baseline is the SEED-AVERAGED v0 OOF gold 0.6173, seed sd
0.0214, margin 0.0427. This is ONE seed, so nothing here is a keep/kill on its own; the
paired within-run comparisons are the trustworthy part. If a policy looks like a real gain,
seeds 1337 and 2024 follow before any verdict.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SEED = 42
N_FOLDS = 5
EPOCHS = 10

META = Path("/kaggle/input/datasets/girishbose/rsna-knee-gf-v0c-meta")
if not META.exists():
    META = Path("/kaggle/input/rsna-knee-gf-v0c-meta")

CACHE = Path("/kaggle/input/datasets/girishbose/rsna-knee-cache-gf-v0/cache_gf_v0")
if not CACHE.exists():
    CACHE = Path("/kaggle/input/rsna-knee-cache-gf-v0/cache_gf_v0")

TRAIN_CSV = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv")
OUT = Path("/kaggle/working/gf_v0c_conv_seed42_5fold")

# Recorded weak-val per epoch from the original 5-epoch seed-42 run (docs/experiments.md).
# Epochs 0-4 of this run should retrace these; small GPU nondeterminism is expected.
V0_SEED42_WEAKVAL = {
    0: [0.695, 0.707, 0.721, 0.718, 0.725],
    1: [0.674, 0.670, 0.682, 0.687, 0.687],
    2: [0.683, 0.720, 0.729, 0.728, 0.743],
    3: [0.703, 0.724, 0.712, 0.714, 0.718],
    4: [0.674, 0.687, 0.675, 0.690, 0.685],
}
SEED_AVERAGED_BASELINE = 0.6173
SINGLE_SEED_MEAN = 0.5918
SEED_SD = 0.0214


def main() -> None:
    assert META.exists(), META
    assert CACHE.exists(), CACHE
    assert TRAIN_CSV.exists(), TRAIN_CSV
    weak = META / "data" / "processed" / "weak_labels_v1.csv"
    assert weak.exists(), weak
    print("cache npz:", len(list(CACHE.glob("*.npz"))), flush=True)
    print(f"seed={SEED} epochs={EPOCHS} folds={N_FOLDS}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(META)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(META / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["DINOV2_REPO"] = str(META / "third_party" / "dinov2")

    for fold in range(N_FOLDS):
        cmd = [
            sys.executable,
            "scripts/train_baseline_fold.py",
            "--config",
            "configs/gf_baseline_v0c.yaml",
            "--train-csv",
            str(TRAIN_CSV),
            "--folds",
            "data/folds/folds_v1.csv",
            "--weak-csv",
            "data/processed/weak_labels_v1.csv",
            "--cache-dir",
            str(CACHE),
            "--fold",
            str(fold),
            "--epochs",
            str(EPOCHS),
            "--freeze-epochs",
            str(EPOCHS),
            "--save-epoch-oof",
            "--out-dir",
            str(OUT),
        ]
        print("Running:", " ".join(cmd), flush=True)
        subprocess.check_call(cmd, env=env)
        for e in range(EPOCHS):
            assert (OUT / f"fold{fold}_ep{e}_oof.csv").exists(), (fold, e)
        print(f"fold {fold} ok", flush=True)

    sys.path.insert(0, str(META / "src"))
    import pandas as pd

    from rsna_knee.constants import LABEL_COLS
    from rsna_knee.epoch_policies import (
        assemble_oof,
        best_weakval_pick,
        final_epoch_pick,
        last_k_pick,
        single_epoch_pick,
    )
    from rsna_knee.metrics import summarize_metrics

    folds = list(range(N_FOLDS))

    train = pd.read_csv(TRAIN_CSV)
    is_gold = train[LABEL_COLS].notna().all(axis=1)
    gold = train.loc[is_gold, ["StudyInstanceUID"] + LABEL_COLS]

    # Weak(+expert) targets, same construction as every previous greenfield run.
    weak_df = pd.read_csv(weak)
    y_weak = train[["StudyInstanceUID"] + LABEL_COLS].copy()
    w = weak_df.set_index("StudyInstanceUID")
    for idx, row in y_weak.iterrows():
        uid = str(row["StudyInstanceUID"])
        if uid not in w.index:
            continue
        wr = w.loc[uid]
        for c in LABEL_COLS:
            if pd.isna(row[c]) and c in wr.index and pd.notna(wr[c]):
                y_weak.at[idx, c] = wr[c]

    histories = {
        f: json.loads((OUT / f"fold{f}_history.json").read_text()) for f in range(N_FOLDS)
    }

    print("\n=== reproduction check: epochs 0-4 weak-val vs recorded 5-epoch seed-42 run ===", flush=True)
    for f in range(N_FOLDS):
        got = [round(float(h["val_macro_auc"]), 3) for h in histories[f][:5]]
        exp = V0_SEED42_WEAKVAL[f]
        drift = max(abs(a - b) for a, b in zip(got, exp))
        print(f"  fold{f}: got {got}  expected {exp}  max|drift| {drift:.3f}", flush=True)

    def oof_for(pick: dict[int, list[int]]) -> pd.DataFrame:
        return assemble_oof(OUT, pick)

    def score(oof: pd.DataFrame) -> tuple[float, float, dict]:
        preds = oof.rename(columns={c: f"{c}__p" for c in LABEL_COLS})
        g = gold.merge(preds, on="StudyInstanceUID", how="inner")
        assert len(g) == len(gold)
        gs = summarize_metrics(
            g[LABEL_COLS].to_numpy(dtype=float),
            g[[f"{c}__p" for c in LABEL_COLS]].to_numpy(dtype=float),
        )
        labs = y_weak.rename(columns={c: f"{c}__y" for c in LABEL_COLS})
        mw = preds.merge(labs, on="StudyInstanceUID", how="inner")
        ws = summarize_metrics(
            mw[[f"{c}__y" for c in LABEL_COLS]].to_numpy(dtype=float),
            mw[[f"{c}__p" for c in LABEL_COLS]].to_numpy(dtype=float),
        )
        return gs["macro_auc"], ws["macro_auc"], gs["per_label_auc"]

    print("\n=== per-epoch true OOF gold-58 curve (is 5 epochs undertrained?) ===", flush=True)
    curve = {}
    for e in range(EPOCHS):
        gm, wm, _ = score(oof_for(single_epoch_pick(folds, e)))
        curve[e] = {"gold": gm, "weak": wm}
        print(f"  epoch {e}: gold {gm:.4f}   weak {wm:.4f}", flush=True)

    policies = {
        "best_weakval_first5_old_recipe": best_weakval_pick(histories, limit=5),
        "best_weakval_all10": best_weakval_pick(histories),
        "final_epoch": final_epoch_pick(folds, EPOCHS),
        "avg_last3": last_k_pick(folds, EPOCHS, 3),
        "avg_last5": last_k_pick(folds, EPOCHS, 5),
    }

    print("\n=== checkpoint-selection policies (same trained models, one seed) ===", flush=True)
    results = {}
    for name, pick in policies.items():
        gm, wm, per = score(oof_for(pick))
        results[name] = {
            "gold58_oof_macro_auc": gm,
            "weak_oof_macro_auc": wm,
            "per_label_auc": per,
            "epochs_used": {str(k): v for k, v in pick.items()},
        }
        print(f"  {name:<32} gold {gm:.4f}   weak {wm:.4f}   epochs {list(pick.values())}", flush=True)

    best_name = max(results, key=lambda k: results[k]["gold58_oof_macro_auc"])
    best_gold = results[best_name]["gold58_oof_macro_auc"]
    old_gold = results["best_weakval_first5_old_recipe"]["gold58_oof_macro_auc"]

    oof_for(policies[best_name]).to_csv(OUT / "oof_all_bestpolicy.csv", index=False)
    bp = oof_for(policies[best_name]).rename(columns={c: f"{c}__p" for c in LABEL_COLS})
    g = gold.merge(bp, on="StudyInstanceUID", how="inner")
    g_out = g[["StudyInstanceUID"]].copy()
    for c in LABEL_COLS:
        g_out[c] = g[f"{c}__p"]
    g_out.to_csv(OUT / "oof_gold58_preds.csv", index=False)

    print("\n=== summary ===", flush=True)
    print(f"old policy (best-on-weak-val, epochs 0-4): gold {old_gold:.4f}", flush=True)
    print(f"best policy here ({best_name}): gold {best_gold:.4f}", flush=True)
    print(f"within-run paired gain: {best_gold - old_gold:+.4f}", flush=True)
    print(
        f"vs seed-averaged v0 baseline {SEED_AVERAGED_BASELINE:.4f} "
        f"(single-seed mean {SINGLE_SEED_MEAN:.4f}, seed sd {SEED_SD:.4f}): "
        f"{best_gold - SEED_AVERAGED_BASELINE:+.4f}",
        flush=True,
    )
    print(
        "NOTE: one seed only. The per-epoch curve and the policy comparison are paired "
        "within this run and are the trustworthy signals. Cross-run deltas are not a verdict.",
        flush=True,
    )

    (OUT / "convergence_report.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "epochs": EPOCHS,
                "per_epoch_curve": curve,
                "policies": results,
                "best_policy": best_name,
                "best_policy_gold": best_gold,
                "old_policy_gold": old_gold,
                "within_run_paired_gain": best_gold - old_gold,
                "seed_averaged_v0_baseline": SEED_AVERAGED_BASELINE,
                "v0_single_seed_mean": SINGLE_SEED_MEAN,
                "v0_seed_sd": SEED_SD,
                "reproduction_check": {
                    str(f): {
                        "got": [float(h["val_macro_auc"]) for h in histories[f][:5]],
                        "expected_seed42_5ep": V0_SEED42_WEAKVAL[f],
                    }
                    for f in range(N_FOLDS)
                },
                "note": (
                    "Single seed. Within-run paired comparisons (epoch curve, selection "
                    "policy) are trustworthy; any cross-run delta must be confirmed at "
                    "seeds 1337 and 2024 before a keep/kill."
                ),
            },
            indent=2,
        )
    )
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
