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

## Next 3 actions
1. Builder `--rad-extra β`: our fused heads blended AFTER the cell-25 calibrator (public heads keep feeding it);
   probe on top of coat-up15-b4 (best 0.945). See PLAN_TO_TOP.md results log 2026-10-02 12:35.
2. DINO fine-tune DONE + gated: holdout-29 pooled 0.8057 → 0.8228 (dev-29 0.8181 → 0.8371; all-58 macro 0.839 → 0.854).
   Weights `girishbose/rsna-knee-dino-ft-fused`. **`coat15-dinox30` submitted (56771185)** (`--dino-extra ds:0.3`: our 20 members run
   after the public DINO phase into a side file, blended at β=0.3 AFTER the cell-25 calibrator) (v2 log verified: 20/20 public + 20/20 ours, blend after calibrator; >90 min runtime guard).
   `coat15-dinoft` (inside-calibrator swap) is a reference run only — do not submit.
3. Per-finding CoAt shift (only lever that has moved LB): e.g. +0.15 where CoAt ≫ transformer on experts.
