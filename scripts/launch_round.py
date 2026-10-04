#!/usr/bin/env python3
"""Launch one label round's models (teachers / students) on parallel pods and add them to the job registry.

    python scripts/launch_round.py --round 1 --labels labels_r1.parquet [--families coat,fa] [--folds 0,1,2,3,4]
    python scripts/launch_round.py --round 0 --labels labels_relabel_8c56cdf0.parquet --families coat --folds 2 --suffix b

Recipes are the clean round-0 ones (ImageNet init, never init:v5): CoAt = n-5bc0a0f45d75 args on dense80,
Family A = runpod/family_a.py 16 epochs. One pod per fold on 48 GB hosts (community 4090s vanished mid-train).
Jobs are named r<round>-<family>-f<fold><suffix>; an existing registry entry for the same round/family/fold is
replaced (relaunch of a vanished pod).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rsna_knee.agent.launcher import launch

REPO = Path(__file__).resolve().parents[1]
REG = REPO / "agent" / "runs" / "label_rounds.json"
PROTO = "--fold_file /workspace/bundle/folds_nonexpert_5.json --holdout /workspace/bundle/expert_holdout.json"
RECIPES = {
    "coat": {"trainer": "train_knee.py", "hours": 7.5, "epochs": 20, "note": "dense80, 384 px",
             "name": "CoAtNet 2.5D",
             "args": "--arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k --res 384 --epochs 20 --bs 8 --k 8 --k_eval 24 "
                     "--bb_lr 3e-05 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet --seed 42 "
                     f"--folds 5 --fold {{fold}} {PROTO} --workers 8"},
    # Cheap 224 px families (forum: 5-fold ResNet50 @224 = 0.954, single CoAtNet @224 = 0.950 on pseudo-labels):
    # diverse teachers for the next round + a test of whether small/low-res models lose anything here.
    "r50": {"trainer": "train_knee.py", "hours": 3.0, "epochs": 20, "note": "dense80, 224 px, k16",
            "name": "ResNet50 2.5D @224",
            "args": "--arch resnet50.a1_in1k --res 224 --epochs 20 --bs 8 --k 16 --k_eval 24 "
                    "--bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --norm imagenet --seed 42 "
                    f"--folds 5 --fold {{fold}} {PROTO} --workers 8"},
    "c224": {"trainer": "train_knee.py", "hours": 3.0, "epochs": 20, "note": "dense80, 224 px, k16",
             "name": "CoAtNet 2.5D @224",
             "args": "--arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224 --epochs 20 --bs 8 --k 16 --k_eval 24 "
                     "--bb_lr 3e-05 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet --seed 42 "
                     f"--folds 5 --fold {{fold}} {PROTO} --workers 8"},
    # Recipe variants of c224 (2026-10-04: c224 k16 beat CoAt-384 k8 on holdout-29, 0.941 vs 0.934) — fold-0 A/Bs.
    "c224k32": {"trainer": "train_knee.py", "hours": 4.0, "epochs": 20, "note": "dense80, 224 px, k32",
                "name": "CoAtNet 2.5D @224 k32",
                "args": "--arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224 --epochs 20 --bs 8 --k 32 --k_eval 32 "
                        "--bb_lr 3e-05 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                        f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    # 2026-10-04: the base recipe has no spatial augmentation and no stochastic depth (only a +-10 % gain jitter).
    # c224reg = c224 + per-window affine/gamma + drop-path 0.1; fold-0 A/B against r1c2240 (same labels, same seed).
    "c224reg": {"trainer": "train_knee.py", "hours": 3.5, "epochs": 20, "note": "dense80, 224 px, k16, affine aug + drop-path",
                "name": "CoAtNet 2.5D @224 + aug",
                "args": "--arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224 --epochs 20 --bs 8 --k 16 --k_eval 24 "
                        "--bb_lr 3e-05 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                        f"--aug_geo 1.0 --drop_path 0.1 --seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    # 2026-10-04 evening: fold-0 recipe sweep on labels_r1, judged by scripts/oof_ruler.py (870 studies, +-0.002) and
    # the experts; only winners go to the LB. Base to beat: r1-c224-f0:r1c2240.
    "c224lr": {"trainer": "train_knee.py", "hours": 3.5, "epochs": 20, "note": "dense80, 224 px, k16, bb_lr 1e-4",
                "name": "CoAtNet @224, backbone lr 1e-4",
                "args": "--arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224 --epochs 20 --bs 8 --k 16 --k_eval 24 "
                        "--bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                        f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    "c224e30": {"trainer": "train_knee.py", "hours": 5.0, "epochs": 30, "note": "dense80, 224 px, k16, 30 epochs",
                "name": "CoAtNet @224, 30 epochs",
                "args": "--arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224 --epochs 30 --bs 8 --k 16 --k_eval 24 "
                        "--bb_lr 3e-05 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                        f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    "c224reg40": {"trainer": "train_knee.py", "hours": 6.5, "epochs": 40, "note": "dense80, 224 px, k16, aug + drop-path, 40 epochs",
                "name": "CoAtNet @224 + aug, 40 epochs",
                "args": "--arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224 --epochs 40 --bs 8 --k 16 --k_eval 24 "
                        "--bb_lr 3e-05 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet --aug_geo 1.0 --drop_path 0.1 "
                        f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    "cnx224": {"trainer": "train_knee.py", "hours": 3.5, "epochs": 20, "note": "dense80, 224 px, k16, convnext_tiny in22k",
                "name": "ConvNeXt-T 2.5D @224",
                "args": "--arch convnext_tiny.fb_in22k_ft_in1k --res 224 --epochs 20 --bs 8 --k 16 --k_eval 24 "
                        "--bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                        f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    # Base loss up-weights positives by (1-prev)/prev capped at 10 (MCL, Fracture x10; Baker's, ACL x5-6), but report
    # YES is the noisy side (experts confirm 44-81 %). c224pw1 = c224 with plain BCE.
    "c224pw1": {"trainer": "train_knee.py", "hours": 3.5, "epochs": 20, "note": "dense80, 224 px, k16, no pos_weight",
                "name": "CoAtNet @224, plain BCE",
                "args": "--arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224 --epochs 20 --bs 8 --k 16 --k_eval 24 "
                        "--bb_lr 3e-05 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                        f"--pos_weight_cap 1 --seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    # NOT runnable: coatnet_rmlp_2_rw_224 has fixed-size relative position bias — any --res other than 224 dies at
    # the first step (r1-c288-f0/f0b, 2026-10-04). A 288 px CoAtNet needs a backbone built with img_size=288.
    "fa": {"trainer": "family_a.py", "hours": 4.0, "epochs": 16, "note": "dense80, region tiles",
           "name": "RegionMIL (ConvNeXt)",
           "args": "--fold {fold} --epochs 16 --grad_ckpt --workers 8"},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--labels", required=True, help="parquet under data/processed")
    ap.add_argument("--families", default="coat,fa")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--suffix", default="", help="job-name suffix for a relaunch (b, c, ...)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    assert (REPO / "data" / "processed" / a.labels).exists(), a.labels
    for fam in a.families.split(","):
        r = RECIPES[fam]
        for fold in (int(x) for x in a.folds.split(",")):
            job, tag = f"r{a.round}-{fam}-f{fold}{a.suffix}", f"r{a.round}{fam}{fold}"
            arms = [(tag, a.labels, r["args"].format(fold=fold))]
            if a.dry_run:
                print(job, arms)
                continue
            # Two sessions drove the same round on 2026-10-03: never start a second pod for a fold by accident.
            have = [j["job"] for j in json.loads(REG.read_text())["jobs"]
                    if j["round"] == a.round and j["family"] == fam and j["fold"] == fold]
            if have and not a.suffix:
                print(f"{job}: already in registry as {have} — skipped (pass --suffix to relaunch)", flush=True)
                continue
            out = launch(job, arms, r["hours"], corpus="dense80", gpu_class="48", trainer=r["trainer"])
            reg = json.loads(REG.read_text())   # re-read: a collector may have marked jobs meanwhile
            reg["jobs"] = [j for j in reg["jobs"]
                           if not (j["round"] == a.round and j["family"] == fam and j["fold"] == fold)]
            reg["jobs"].append({"job": job, "pod": out["pod"], "tag": tag, "family": fam, "fold": fold,
                                "round": a.round, "epochs": r["epochs"], "price": out["costPerHr"],
                                "launched": time.strftime("%Y-%m-%dT%H:%M:%S")})
            if reg["round"] != a.round:   # the page and collector follow the newest round
                reg["round"] = a.round
            if fam not in [f["key"] for f in reg["families"]]:
                reg["families"].append({"key": fam, "name": r["name"], "note": r["note"], "globs": []})
            for f in reg["families"]:
                if f["key"] == fam and a.round > 0:
                    f["globs"] = [f"r{a.round}-{fam}-f*/*"]
                    f["note"] = f"{RECIPES[fam]['note']}, {a.labels.removesuffix('.parquet')}"
            REG.write_text(json.dumps(reg, indent=1))
            print(job, out["pod"], out["costPerHr"], flush=True)


if __name__ == "__main__":
    main()
