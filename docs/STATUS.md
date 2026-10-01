# STATUS

Last updated: 2026-10-01 (~09:30 IST). See docs/HANDOFF.md for the full picture.

## Leaderboard
| ref | kernel | public |
|---|---|---:|
| 56651119 | D4 + 5bc0 tail w=0.2 | 0.941 |
| 56665556 | swap m6edbb53fbab → v5 | 0.943 |
| 56680743 | swap m0b16e9a09f1 → v5 | 0.943 |
| **56704765** | **swap soupbestof4 → v5** | **0.944 (rank 258/4660)** |

## Best models (AUC on 58 experts)
- soupbestof4 0.9365 (in LB 0.944); bestof_v1 k=8 members 0.9330; public v5 0.9256.
- k=16 bestof members 0.9266 overall but **PF OA 0.882** (vs 0.855) → PF OA specialist.
- Weak findings everywhere: Lateral Meniscus, Lateral OA, PF OA, Synovitis, Contusion.

## Live
- No pods running (m-7ee65c9063 collecting its 2nd arm). Strategist idle-capped by policy.
- Policy: self-training closed (`agent/runs/policy.json`); 2 seeds max per recipe; 2 pods max.

## Money
- Campaign ≈ $80 of $300. **RunPod ≈ $9.7 — top up ~$20 before more GPU.** DeepSeek ≈ $30.

## Next 3 actions
1. `--label_weights` (2–3× on the 5 weak findings) → 2 bestof_v1 k=8 v5 members.
2. Per-finding tail weights in build_ensemble_kernel (weak findings only) with specialist leg (k16 for PF OA).
3. Soup per-finding winners; dual finals by 2026-10-16.
