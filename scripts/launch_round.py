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
    # 2026-10-04 night: forum topic 743148 (teams at 0.94 single fold / 5-fold): "heavy augmentation and regularization",
    # 50 epochs with the best epoch at 28. Our aug A/B stopped at 20 epochs. c224hvy40 = c224reg40 + left-right mirror
    # + coarse dropout + drop-path 0.2.
    "c224hvy40": {"trainer": "train_knee.py", "hours": 6.5, "epochs": 40,
                  "note": "dense80, 224 px, k16, affine + mirror + erase + drop-path 0.2, 40 epochs",
                  "name": "CoAtNet @224 + heavy aug, 40 epochs",
                  "args": "--arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224 --epochs 40 --bs 8 --k 16 --k_eval 24 "
                          "--bb_lr 3e-05 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                          "--aug_geo 1.0 --aug_flip 0.5 --aug_erase 0.25 --drop_path 0.2 "
                          f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    # Forum topic 745214 (SpeedSci, #122): the same team's EffNet / ResNet50 score 0.940 on LB at ~0.89 on gold-58, their
    # CoAtNet 0.926 at 0.915 on gold - gold-58 overrates the public CoAt recipe. CNNs at the cache's native 336 px with
    # the heavy-aug 40-epoch recipe; judged by oof_ruler.py and an own-alone LB probe, never by gold.
    "effv2s336": {"trainer": "train_knee.py", "hours": 5.0, "epochs": 40,
                  "note": "dense80, 336 px, k16, EfficientNetV2-S in21k, heavy aug, 40 epochs",
                  "name": "EffNetV2-S 2.5D @336 + heavy aug",
                  "args": "--arch tf_efficientnetv2_s.in21k_ft_in1k --res 336 --epochs 40 --bs 8 --k 16 --k_eval 24 "
                          "--bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                          "--aug_geo 1.0 --aug_flip 0.5 --aug_erase 0.25 --drop_path 0.2 "
                          f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    "effb0336": {"trainer": "train_knee.py", "hours": 4.0, "epochs": 40,
                 "note": "dense80, 336 px, k16, EfficientNet-B0 noisy-student, heavy aug, 40 epochs",
                 "name": "EffNet-B0 2.5D @336 + heavy aug",
                 "args": "--arch tf_efficientnet_b0.ns_jft_in1k --res 336 --epochs 40 --bs 8 --k 16 --k_eval 24 "
                         "--bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                         "--aug_geo 1.0 --aug_flip 0.5 --aug_erase 0.25 --drop_path 0.1 "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    # Self-supervised backbone family (forum: SpeedSci's DINOv2 0.942 on LB; Md Kishor's DINOv2 ViT-S at 336 px fully
    # fine-tuned). Our Aug-Sep DINOv2 work kept the backbone frozen or unfroze it at a head-sized lr (-0.114); this is a
    # full fine-tune at 2e-5 inside the current pipeline with the heavy-aug recipe.
    "dinos336": {"trainer": "train_knee.py", "hours": 6.0, "epochs": 40,
                 "note": "dense80, 336 px, k16, DINOv2 ViT-S/14 full fine-tune, heavy aug, 40 epochs",
                 "name": "DINOv2 ViT-S 2.5D @336 + heavy aug",
                 "args": "--arch vit_small_patch14_dinov2.lvd142m --res 336 --epochs 40 --bs 8 --k 16 --k_eval 24 "
                         "--bb_lr 2e-05 --head_lr 0.001 --wd 0.05 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                         "--aug_geo 1.0 --aug_flip 0.5 --aug_erase 0.25 --drop_path 0.1 "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    # 2026-10-05 (user: run wide, find the best system): more backbone families in the heavy-aug 40-epoch recipe, as
    # ensemble members. Fold 0 first; a family goes to 5 folds when it is at par with c224 on the ruler.
    "sex50336": {"trainer": "train_knee.py", "hours": 5.0, "epochs": 40,
                 "note": "dense80, 336 px, k16, SE-ResNeXt50, heavy aug, 40 epochs",
                 "name": "SE-ResNeXt50 2.5D @336 + heavy aug",
                 "args": "--arch seresnext50_32x4d.racm_in1k --res 336 --epochs 40 --bs 8 --k 16 --k_eval 24 "
                         "--bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                         "--aug_geo 1.0 --aug_flip 0.5 --aug_erase 0.25 --drop_path 0.2 "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    "swint224": {"trainer": "train_knee.py", "hours": 5.0, "epochs": 40,
                 "note": "dense80, 224 px, k16, Swin-T in22k, heavy aug, 40 epochs",
                 "name": "Swin-T 2.5D @224 + heavy aug",
                 "args": "--arch swin_tiny_patch4_window7_224.ms_in22k_ft_in1k --res 224 --epochs 40 --bs 8 --k 16 --k_eval 24 "
                         "--bb_lr 5e-05 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                         "--aug_geo 1.0 --aug_flip 0.5 --aug_erase 0.25 --drop_path 0.2 "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    "cnxs336": {"trainer": "train_knee.py", "hours": 6.0, "epochs": 40,
                 "note": "dense80, 336 px, k16, ConvNeXt-S in12k, heavy aug, 40 epochs",
                 "name": "ConvNeXt-S 2.5D @336 + heavy aug",
                 "args": "--arch convnext_small.in12k_ft_in1k --res 336 --epochs 40 --bs 8 --k 16 --k_eval 24 "
                         "--bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                         "--aug_geo 1.0 --aug_flip 0.5 --aug_erase 0.25 --drop_path 0.2 "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    # 2026-10-05: ConvNeXt-S in12k is the best single arm (fold 0 said +0.0070). Same recipe on larger and on
    # masked-autoencoder-pretrained (ConvNeXtV2 fcmae) variants; fold 0 first.
    "cnxv2t336": {"trainer": "train_knee.py", "hours": 6.0, "epochs": 40,
                 "note": "dense80, 336 px, k16, ConvNeXtV2-T fcmae in22k, heavy aug, 40 epochs",
                 "name": "ConvNeXtV2-T (fcmae) 2.5D @336 + heavy aug",
                 "args": "--arch convnextv2_tiny.fcmae_ft_in22k_in1k --res 336 --epochs 40 --bs 8 --k 16 --k_eval 24 "
                         "--bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                         "--aug_geo 1.0 --aug_flip 0.5 --aug_erase 0.25 --drop_path 0.2 "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    "cnxb336": {"trainer": "train_knee.py", "hours": 14.0, "epochs": 40,
                 "note": "dense80, 336 px, k16, ConvNeXt-B in22k, heavy aug, 40 epochs",
                 "name": "ConvNeXt-B 2.5D @336 + heavy aug",
                 "args": "--arch convnext_base.fb_in22k_ft_in1k --res 336 --epochs 40 --bs 8 --k 16 --k_eval 24 "
                         "--bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                         "--aug_geo 1.0 --aug_flip 0.5 --aug_erase 0.25 --drop_path 0.3 "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    "cnxv2b336": {"trainer": "train_knee.py", "hours": 14.0, "epochs": 40,
                 "note": "dense80, 336 px, k16, ConvNeXtV2-B fcmae in22k, heavy aug, 40 epochs",
                 "name": "ConvNeXtV2-B (fcmae) 2.5D @336 + heavy aug",
                 "args": "--arch convnextv2_base.fcmae_ft_in22k_in1k --res 336 --epochs 40 --bs 8 --k 16 --k_eval 24 "
                         "--bb_lr 5e-05 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                         "--aug_geo 1.0 --aug_flip 0.5 --aug_erase 0.25 --drop_path 0.3 "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    # Stage 3 (queue pod only: the fastMRI cache lives on its volume): competition studies with labels_r3 + ~10k
    # fastMRI studies with teacher targets (labels_r3x). An epoch is ~3.9x longer, so 12-14 epochs give about the
    # same number of updates as 40 epochs on the competition studies alone.
    "c224hvyx": {"trainer": "train_knee.py", "hours": 8.0, "epochs": 12,
                 "note": "dense80 + fastMRI cache, 224 px, k16, CoAtNet, heavy aug, 12 epochs, ext weight 0.5",
                 "name": "CoAtNet @224 heavy + fastMRI",
                 "args": "--arch coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k --res 224 --epochs 12 --bs 8 --k 16 --k_eval 24 "
                         "--bb_lr 3e-05 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                         "--aug_geo 1.0 --aug_flip 0.5 --aug_erase 0.25 --drop_path 0.2 "
                         "--ext_dir /workspace/fastmri/dense80 --ext_weight 0.5 "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    "effv2sx": {"trainer": "train_knee.py", "hours": 8.0, "epochs": 14,
                 "note": "dense80 + fastMRI cache, 336 px, k16, EfficientNetV2-S, heavy aug, 14 epochs, ext weight 0.5",
                 "name": "EffNetV2-S @336 heavy + fastMRI",
                 "args": "--arch tf_efficientnetv2_s.in21k_ft_in1k --res 336 --epochs 14 --bs 8 --k 16 --k_eval 24 "
                         "--bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                         "--aug_geo 1.0 --aug_flip 0.5 --aug_erase 0.25 --drop_path 0.2 "
                         "--ext_dir /workspace/fastmri/dense80 --ext_weight 0.5 "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    "cnxsx": {"trainer": "train_knee.py", "hours": 8.0, "epochs": 12,
                 "note": "dense80 + fastMRI cache, 336 px, k16, ConvNeXt-S, heavy aug, 12 epochs, ext weight 0.5",
                 "name": "ConvNeXt-S @336 heavy + fastMRI",
                 "args": "--arch convnext_small.in12k_ft_in1k --res 336 --epochs 12 --bs 8 --k 16 --k_eval 24 "
                         "--bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                         "--aug_geo 1.0 --aug_flip 0.5 --aug_erase 0.25 --drop_path 0.2 "
                         "--ext_dir /workspace/fastmri/dense80 --ext_weight 0.5 "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    "cnxsx1": {"trainer": "train_knee.py", "hours": 8.0, "epochs": 12,
                 "note": "dense80 + fastMRI cache, 336 px, k16, ConvNeXt-S, heavy aug, 12 epochs, ext weight 1.0",
                 "name": "ConvNeXt-S @336 heavy + fastMRI (ext weight 1)",
                 "args": "--arch convnext_small.in12k_ft_in1k --res 336 --epochs 12 --bs 8 --k 16 --k_eval 24 "
                         "--bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet "
                         "--aug_geo 1.0 --aug_flip 0.5 --aug_erase 0.25 --drop_path 0.2 "
                         "--ext_dir /workspace/fastmri/dense80 --ext_weight 1.0 "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    # 2026-10-06: every model so far reads a study as an unordered bag of windows drawn anywhere in the 80-slot volume
    # (no plane / sequence / position information, ~10 % of the windows straddle two series). These arms keep the
    # ConvNeXt-S heavy recipe and add series-aware ordered sampling plus a slot-aware head (train_knee.py --head /
    # --sample); every new part starts at zero, i.e. from the plain head. Twin: r4-cnxs336-f0 (labels_r4).
    "cnxsseq": {"trainer": "train_knee.py", "hours": 6.0, "epochs": 40,
                 "note": "dense80, 336 px, k16, ConvNeXt-S, heavy aug, 40 epochs, slot embeddings + transformer over the windows",
                 "name": "ConvNeXt-S @336 seq head",
                 "args": "--arch convnext_small.in12k_ft_in1k --res 336 --epochs 40 --bs 8 --k 16 --k_eval 24 --bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet --aug_geo 1.0 --aug_flip 0.5 --aug_erase 0.25 --drop_path 0.2 --head seq --sample strat "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    "cnxstok": {"trainer": "train_knee.py", "hours": 6.0, "epochs": 40,
                 "note": "dense80, 336 px, k16, ConvNeXt-S, heavy aug, 40 epochs, slot embeddings + per-finding spatial attention",
                 "name": "ConvNeXt-S @336 tok head",
                 "args": "--arch convnext_small.in12k_ft_in1k --res 336 --epochs 40 --bs 8 --k 16 --k_eval 24 --bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet --aug_geo 1.0 --aug_flip 0.5 --aug_erase 0.25 --drop_path 0.2 --head tok --sample strat "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    "cnxsst": {"trainer": "train_knee.py", "hours": 6.0, "epochs": 40,
                 "note": "dense80, 336 px, k16, ConvNeXt-S, heavy aug, 40 epochs, slot embeddings + transformer + spatial attention",
                 "name": "ConvNeXt-S @336 seqtok head",
                 "args": "--arch convnext_small.in12k_ft_in1k --res 336 --epochs 40 --bs 8 --k 16 --k_eval 24 --bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet --aug_geo 1.0 --aug_flip 0.5 --aug_erase 0.25 --drop_path 0.2 --head seqtok --sample strat "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    "cnxsseq224": {"trainer": "train_knee.py", "hours": 6.0, "epochs": 40,
                 "note": "dense80, 224 px, k26, ConvNeXt-S, heavy aug, 40 epochs, 224 px, every slot each step (k26), transformer head",
                 "name": "ConvNeXt-S @224 seq head",
                 "args": "--arch convnext_small.in12k_ft_in1k --res 224 --epochs 40 --bs 8 --k 26 --k_eval 40 --bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet --aug_geo 1.0 --aug_flip 0.5 --aug_erase 0.25 --drop_path 0.2 --head seq --sample strat "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    # 2026-10-06: canonical side. Orientation is uniform in the DICOMs (geom dump: every sagittal / coronal / axial
    # series has the same row, column and normal direction), so the only left-right nuisance is the knee side: half the
    # studies are left knees. `--canon` mirrors left knees (side from the stack's x position, agent/protocol/
    # knee_sides.json), which puts medial / lateral on one image side and one end of the sagittal stack. No mirror aug.
    "cnxsseqc": {"trainer": "train_knee.py", "hours": 6.0, "epochs": 40,
                 "note": "dense80, 336 px, k16, ConvNeXt-S, heavy aug without mirror, 40 epochs, seq head, canonical side",
                 "name": "ConvNeXt-S @336 seq head, canonical side",
                 "args": "--arch convnext_small.in12k_ft_in1k --res 336 --epochs 40 --bs 8 --k 16 --k_eval 24 --bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet --aug_geo 1.0 --aug_flip 0 --aug_erase 0.25 --drop_path 0.2 --head seq --sample strat --canon /workspace/bundle/knee_sides.json "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    "cnxstokc": {"trainer": "train_knee.py", "hours": 6.0, "epochs": 40,
                 "note": "dense80, 336 px, k16, ConvNeXt-S, heavy aug without mirror, 40 epochs, tok head, canonical side",
                 "name": "ConvNeXt-S @336 tok head, canonical side",
                 "args": "--arch convnext_small.in12k_ft_in1k --res 336 --epochs 40 --bs 8 --k 16 --k_eval 24 --bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet --aug_geo 1.0 --aug_flip 0 --aug_erase 0.25 --drop_path 0.2 --head tok --sample strat --canon /workspace/bundle/knee_sides.json "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    "cnxsstc": {"trainer": "train_knee.py", "hours": 6.0, "epochs": 40,
                 "note": "dense80, 336 px, k16, ConvNeXt-S, heavy aug without mirror, 40 epochs, seqtok head, canonical side",
                 "name": "ConvNeXt-S @336 seqtok head, canonical side",
                 "args": "--arch convnext_small.in12k_ft_in1k --res 336 --epochs 40 --bs 8 --k 16 --k_eval 24 --bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet --aug_geo 1.0 --aug_flip 0 --aug_erase 0.25 --drop_path 0.2 --head seqtok --sample strat --canon /workspace/bundle/knee_sides.json "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
    "cnxsseq224c": {"trainer": "train_knee.py", "hours": 6.0, "epochs": 40,
                 "note": "dense80, 224 px, k26, ConvNeXt-S, heavy aug without mirror, 40 epochs, seq head, canonical side",
                 "name": "ConvNeXt-S @224 seq head, canonical side",
                 "args": "--arch convnext_small.in12k_ft_in1k --res 224 --epochs 40 --bs 8 --k 26 --k_eval 40 --bb_lr 1e-04 --head_lr 0.001 --wd 0.02 --freeze_blocks 0 --grad_ckpt --norm imagenet --aug_geo 1.0 --aug_flip 0 --aug_erase 0.25 --drop_path 0.2 --head seq --sample strat --canon /workspace/bundle/knee_sides.json "
                         f"--seed 42 --folds 5 --fold {{fold}} {PROTO} --workers 8"},
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
