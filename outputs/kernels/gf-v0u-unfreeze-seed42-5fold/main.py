"""gf_v0u stage 1: careful backbone unfreeze after frozen warmup (paired, one seed).

Reuses rsna-knee-gf-v0c-meta (has --save-epoch-oof + epoch_policies). No new dataset.
Recipe: same cache_gf_v0 / weak_v1 / DINOv2-S / seed 42 / lr 3e-4.
  epochs 0-4: backbone frozen (retraces gf_v0 / v0c)
  epochs 5-7: full backbone unfrozen at lr * 0.05

Pre-registered gates (within-run, paired; NOT a multi-seed macro keep):
  COLLAPSE if max(gold ep5-7) < max(gold ep0-4) - 0.02  -> kill, no multi-seed
  PROMISING if max(gold ep5-7) >= max(gold ep0-4) + 0.02 -> seeds 1337+2024
  else INCONCLUSIVE -> kill (not worth 3-seed spend)

Reopens DECISIONS 2026-08-12 only under honest paired measurement.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SEED = 42
N_FOLDS = 5
EPOCHS = 8
FREEZE_EPOCHS = 5
UNFREEZE_LR_MULT = 0.05
COLLAPSE_MARGIN = 0.02
PROMISE_MARGIN = 0.02

META = Path("/kaggle/input/datasets/girishbose/rsna-knee-gf-v0c-meta")
if not META.exists():
    META = Path("/kaggle/input/rsna-knee-gf-v0c-meta")

CACHE = Path("/kaggle/input/datasets/girishbose/rsna-knee-cache-gf-v0/cache_gf_v0")
if not CACHE.exists():
    CACHE = Path("/kaggle/input/rsna-knee-cache-gf-v0/cache_gf_v0")

TRAIN_CSV = Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv")
OUT = Path("/kaggle/working/gf_v0u_unfreeze_seed42_5fold")

# Epochs 0-4 should retrace the recorded seed-42 5-epoch weak-val (pairing check).
V0_SEED42_WEAKVAL = {
    0: [0.695, 0.707, 0.721, 0.718, 0.725],
    1: [0.674, 0.670, 0.682, 0.687, 0.687],
    2: [0.683, 0.720, 0.729, 0.728, 0.743],
    3: [0.703, 0.724, 0.712, 0.714, 0.718],
    4: [0.674, 0.687, 0.675, 0.690, 0.685],
}


def main() -> None:
    assert META.exists(), META
    assert CACHE.exists(), CACHE
    assert TRAIN_CSV.exists(), TRAIN_CSV
    weak = META / "data" / "processed" / "weak_labels_v1.csv"
    assert weak.exists(), weak
    print(
        f"seed={SEED} epochs={EPOCHS} freeze={FREEZE_EPOCHS} "
        f"unfreeze_lr_mult={UNFREEZE_LR_MULT}",
        flush=True,
    )
    print("cache npz:", len(list(CACHE.glob("*.npz"))), flush=True)

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
            "configs/gf_baseline_v0.yaml",
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
            str(FREEZE_EPOCHS),
            "--unfreeze-lr-mult",
            str(UNFREEZE_LR_MULT),
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
    from rsna_knee.epoch_policies import assemble_oof, single_epoch_pick
    from rsna_knee.metrics import summarize_metrics

    folds = list(range(N_FOLDS))
    train = pd.read_csv(TRAIN_CSV)
    is_gold = train[LABEL_COLS].notna().all(axis=1)
    gold = train.loc[is_gold, ["StudyInstanceUID"] + LABEL_COLS]

    histories = {
        f: json.loads((OUT / f"fold{f}_history.json").read_text()) for f in range(N_FOLDS)
    }
    print("\n=== reproduction check: epochs 0-4 weak-val vs recorded seed-42 ===", flush=True)
    for f in range(N_FOLDS):
        got = [round(float(h["val_macro_auc"]), 3) for h in histories[f][:5]]
        exp = V0_SEED42_WEAKVAL[f]
        drift = max(abs(a - b) for a, b in zip(got, exp))
        print(f"  fold{f}: got {got}  expected {exp}  max|drift| {drift:.3f}", flush=True)

    def score_pick(pick: dict[int, list[int]]) -> tuple[float, float]:
        oof = assemble_oof(OUT, pick)
        preds = oof.rename(columns={c: f"{c}__p" for c in LABEL_COLS})
        g = gold.merge(preds, on="StudyInstanceUID", how="inner")
        assert len(g) == len(gold)
        gs = summarize_metrics(
            g[LABEL_COLS].to_numpy(dtype=float),
            g[[f"{c}__p" for c in LABEL_COLS]].to_numpy(dtype=float),
        )
        # weak smoke from histories only for the curve; skip full weak OOF rebuild here
        return float(gs["macro_auc"]), gs["per_label_auc"]

    print("\n=== per-epoch true OOF gold-58 ===", flush=True)
    curve = {}
    for e in range(EPOCHS):
        gm, per = score_pick(single_epoch_pick(folds, e))
        phase = "frozen" if e < FREEZE_EPOCHS else "unfrozen"
        curve[e] = {"gold": gm, "per_label_auc": per, "phase": phase}
        print(f"  epoch {e} [{phase}]: gold {gm:.4f}", flush=True)

    frozen_gold = [curve[e]["gold"] for e in range(FREEZE_EPOCHS)]
    unfrozen_gold = [curve[e]["gold"] for e in range(FREEZE_EPOCHS, EPOCHS)]
    frozen_max = max(frozen_gold)
    unfrozen_max = max(unfrozen_gold)
    delta = unfrozen_max - frozen_max

    if unfrozen_max < frozen_max - COLLAPSE_MARGIN:
        verdict = "COLLAPSE"
    elif unfrozen_max >= frozen_max + PROMISE_MARGIN:
        verdict = "PROMISING"
    else:
        verdict = "INCONCLUSIVE"

    print("\n=== STAGE1 GATE ===", flush=True)
    print(f"frozen max gold (ep0-{FREEZE_EPOCHS-1}): {frozen_max:.4f}", flush=True)
    print(f"unfrozen max gold (ep{FREEZE_EPOCHS}-{EPOCHS-1}): {unfrozen_max:.4f}", flush=True)
    print(f"delta: {delta:+.4f}  verdict: {verdict}", flush=True)
    if verdict == "PROMISING":
        print("Next: seeds 1337 + 2024 for seed-averaged keep/kill.", flush=True)
    else:
        print("Kill without multi-seed.", flush=True)

    # Save best-phase OOF for later A/B tools
    best_ep = max(range(EPOCHS), key=lambda e: curve[e]["gold"])
    oof_best = assemble_oof(OUT, single_epoch_pick(folds, best_ep))
    oof_best.to_csv(OUT / "oof_all_best_epoch.csv", index=False)
    preds = oof_best.rename(columns={c: f"{c}__p" for c in LABEL_COLS})
    g = gold.merge(preds, on="StudyInstanceUID", how="inner")
    g_out = g[["StudyInstanceUID"]].copy()
    for c in LABEL_COLS:
        g_out[c] = g[f"{c}__p"]
    g_out.to_csv(OUT / "oof_gold58_preds.csv", index=False)

    (OUT / "unfreeze_report.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "epochs": EPOCHS,
                "freeze_epochs": FREEZE_EPOCHS,
                "unfreeze_lr_mult": UNFREEZE_LR_MULT,
                "per_epoch_curve": {
                    str(e): {"gold": curve[e]["gold"], "phase": curve[e]["phase"]}
                    for e in range(EPOCHS)
                },
                "frozen_max_gold": frozen_max,
                "unfrozen_max_gold": unfrozen_max,
                "delta_unfrozen_minus_frozen": delta,
                "collapse_margin": COLLAPSE_MARGIN,
                "promise_margin": PROMISE_MARGIN,
                "verdict": verdict,
                "best_epoch": best_ep,
                "best_epoch_gold": curve[best_ep]["gold"],
                "best_epoch_per_label": curve[best_ep]["per_label_auc"],
                "note": (
                    "Paired within-run unfreeze test. Epochs 0-4 should match gf_v0. "
                    "Verdict is stage1 gate only; macro keep needs 3-seed average."
                ),
            },
            indent=2,
        )
    )
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
