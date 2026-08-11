# Rank-1 roadmap (RSNA Knee Abnormality Detection)

Goal: **macro AUC** competitive with public LB leaders (~0.94+), plus a separate efficiency final.
Current floor: frozen DINOv2-S + weak_labels_v1 ≈ **0.72 OOF**. Do not submit until OOF clearly lifts.

## Why ~0.72 is not enough
- Public top ≈ 0.94 — gap is supervision + capacity, not “one more fold of the same recipe.”
- v1 weak labels were EN+ES-heavy while **French is the largest report language**.
- Cache is thin (3 series × 12 slices × 224); leaders likely use more series/slices and stronger backbones.
- Only 58 expert studies — weak labels dominate training noise.

## Levers ordered by expected ROI

### 1. Multilingual weak labels v2 (ship now)
- FR/DE/PT/NL patterns + EN/ES; expert override; export `weak_labels_v2.csv`.
- Success: higher expert-audit F1/recall vs v1; more studies with ≥1 supervised label.
- Retrain **same** frozen S recipe on v2 → isolate label lift before changing backbone.

### 2. DINOv2-B + staged train (main track)
- Config: `configs/main_dinov2_b.yaml` (freeze 4ep → unfreeze lr×0.05, `pos_weight=2`).
- Bundle public `dinov2_vitb14` weights offline.
- Optional: richer cache (4×16×224) as `cache_v2` only after v2-label S run shows lift.

### 3. Rare-label / confidence weighting
- `pos_weight` + existing confidence mask already in trainer.
- Next: per-label pos weights from prevalence; optional Focal/Asymmetric loss if rare labels lag.

### 4. Expert fine-tune stage
- After weak pretrain: short head (then gentle backbone) fine-tune on 58 experts with study-level CV / leave-some-out.
- Do not train experts-only from scratch.

### 5. OOF blend then one LB probe
- 5-fold OOF blend (S+v2 and/or B); calibrate if needed.
- Submit **only** if OOF ≫ 0.72 (target mid-0.8+ before first probe). Log every submit in `experiments.md`.

### 6. Efficiency final (parallel late)
- Distill B → S student (`configs/efficiency_student.yaml`); minimize score formula; never burn both finals on twins.

## Explicitly defer / avoid
- Aggressive full unfreeze at high LR (already collapsed).
- Report text at test time (impossible / leakage design).
- Chasing public LB before OOF win.
- Private pretrained weights (must be public + offline-bundled).

## Immediate session checklist
1. Audit + export `weak_labels_v2.csv`
2. Repackage/upload `rsna-knee-code` (+ Vit-B weights dataset if training B)
3. Kaggle: 5-fold frozen S on **v2** labels (A/B vs 0.719)
4. If lift: start B fold0 with gentle unfreeze; else iterate FR patterns / OA phrases
5. Update STATUS with new OOF before any submit
