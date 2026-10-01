# STATUS

Last updated: 2026-10-02 (~05:00 IST). Plan: docs/PLAN_TO_TOP.md (results log + next build).

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
| 56760305 | coat15-radfused: + fused-label RadImageNet heads | pending |
| (auto 05:30 IST) | LEARNING coat30-radfused: CoAt +0.30 + fused heads | pending |

## Best models (AUC on 58 experts)
- soupbestof4 0.9365 (in LB 0.944); bestof_v1 k=8 members 0.9330; public v5 0.9256.
- k=16 bestof members 0.9266 overall but **PF OA 0.882** (vs 0.855) → `soupk16` (2-seed soup, uploaded
  `girishbose/rsna-knee-soup-k16`, not yet expert-evaluated) = PF OA specialist leg.
- Weak findings everywhere: Lateral Meniscus, Lateral OA, PF OA, Synovitis, Contusion.

## Live
- Fused-label RadImageNet heads beat upstream (E13 0.8745 vs 0.8576; main 0.8594 vs 0.8543) → dataset
  `girishbose/rsna-knee-rad-heads-fused`, wired via `--rad-heads`. Background job submits coat30 probe after reset and
  records both scores in submits.json.
- No RunPod pods; strategist propose-only. Next build: DINO warm-start fine-tune (design in PLAN_TO_TOP.md).

## Money
- Campaign ≈ $85 of $300. **RunPod $4.10 at 20:30 IST**; user tops up 2026-10-02. DeepSeek ≈ $30.

## Submission policy
Only submit candidates expected to beat 0.944 (or explicit learning probes). Gate ≤5/day.

## Next 3 actions
1. Read 56760305 / coat30 scores; keep the better coat shift as the new base.
2. Build + run the DINO fine-tune kernel (PLAN_TO_TOP.md "Next build"); gate on gold OOF > 0.8425; wire into stack.
3. Locate the A5 trainer; same recipe. RunPod top-up → v8/v10 fine-tunes only if stack evidence supports it.
