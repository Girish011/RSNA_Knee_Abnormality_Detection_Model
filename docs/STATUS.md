# STATUS

Last updated: 2026-10-02 (~23:30 IST). Plan: docs/PLAN_TO_TOP.md (results log + next build).

## Leaderboard
| ref | kernel | public |
|---|---|---:|
| 56651119 | D4 + 5bc0 tail w=0.2 | 0.941 |
| 56665556 | swap m6edbb53fbab → v5 | 0.943 |
| 56680743 | swap m0b16e9a09f1 → v5 | 0.943 |
| **56704765** | **swap soupbestof4 → v5** | **0.944 (rank 258/4660)** |
| 56744344 | + PF OA ← soupk16 w0.4 | 0.944 |
| 56744346 | + PF OA, Synovitis ← soupk16 w0.35 | 0.944 |
| 56754652 | LEARNING raptor-all-b4: soup4 in v5+v8+v10 | 0.944 |
| **56754655** | **coat-up15-b4: soup4 swap + CoAt weight +0.15** | **0.945 (best)** |
| 56760305 | coat15-radfused: + fused-label RadImageNet heads | 0.944 |
| 56761295 | LEARNING coat30-radfused: CoAt +0.30 + fused heads | 0.944 |
| 56771185 | coat15-dinox30: base + fine-tuned DINO β0.3 after calibrator | 0.944 |
| 56779149 | LEARNING coat15-radexfx: best + Fracture in Rad blend (d4-blend tweak) | pending |

## Best models (AUC on 58 experts)
- soupbestof4 0.9365 (in LB 0.944); bestof_v1 k=8 members 0.9330; public v5 0.9256.
- k=16 bestof members 0.9266 overall but **PF OA 0.882** (vs 0.855) → `soupk16` (2-seed soup, uploaded
  `girishbose/rsna-knee-soup-k16`, not yet expert-evaluated) = PF OA specialist leg.
- Weak findings everywhere: Lateral Meniscus, Lateral OA, PF OA, Synovitis, Contusion.

## Live
- DINO fine-tune finished (3.3 h free T4x2, 20/20, pixel corr 0.998). Builder `scripts/build_dino_ft_kernel.py`, gate `scripts/eval_dino_ft.py`.
- Fused RadImageNet heads: gold OOF +0.017 (E13) but LB 0.944 vs 0.945 → **gold-58 gains do not transfer**.
  Gold-58 is contaminated for models trained on our labels (bestof selection + fusion fit on experts): our 5-fold
  CoAt alone scores 0.931 on gold-58 but our own-weights LB was 0.920. LB probes are the judge.
- No RunPod pods. **Strategist paused** (policy `strategist: paused`, user 2026-10-02); loop still monitors pods/members.
## Money
- Campaign ≈ $85 of $300. **RunPod ~$20 (topped up 2026-10-02)**; Claude may spend it on new training tries that need GPU capacity; default $0 until a lever is LB-supported. DeepSeek ≈ $30.

## Submission policy
Only submit candidates expected to beat 0.944 (or explicit learning probes). Gate ≤5/day.

## Family A (own model, 2026-10-03)
- `runpod/family_a.py` + rulers built. Fold-0 `fa-f0-v2` FAILED rulers (weak5 vs pilk 0.842 vs CoAt 0.884; blend lowers). Fused-label AUC is not a ruler.
- Family B (fastMRI+ encoder init) A/B fb-f0-v2: no gain on report labels (weak5 pilk 0.833 vs 0.842); holdout-29 +0.03 (noisy). DICOMs skipped per rule. RunPod ≈ $5 of $20 spent.
- 56779149 score not yet recorded (Kaggle CLI call was blocked in-session; user to run `submit_gate.py record`).

## Next 3 actions
1. Record 56779149 (coat15-radexfx, LEARNING: Fracture into the Rad blend as in public d4-blend 0.946).
2. **Own-model track (PLAN_TO_TOP 2026-10-02 night):** Family A weak-five specialists on dense80 (RunPod, ~$20), final-level blend.
3. User registers for fastMRI (docs/EXTERNAL_DATA.md) → Family B pretraining when links land in ~/.rsna_ext/fastmri_links.txt.
