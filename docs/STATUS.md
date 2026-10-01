# STATUS

Last updated: 2026-10-01 (~10:30 IST). See docs/HANDOFF.md for the full picture.

## Leaderboard
| ref | kernel | public |
|---|---|---:|
| 56651119 | D4 + 5bc0 tail w=0.2 | 0.941 |
| 56665556 | swap m6edbb53fbab → v5 | 0.943 |
| 56680743 | swap m0b16e9a09f1 → v5 | 0.943 |
| **56704765** | **swap soupbestof4 → v5** | **0.944 (rank 258/4660)** |

## Best models (AUC on 58 experts)
- soupbestof4 0.9365 (in LB 0.944); bestof_v1 k=8 members 0.9330; public v5 0.9256.
- k=16 bestof members 0.9266 overall but **PF OA 0.882** (vs 0.855) → `soupk16` (2-seed soup, uploaded
  `girishbose/rsna-knee-soup-k16`, not yet expert-evaluated) = PF OA specialist leg.
- Weak findings everywhere: Lateral Meniscus, Lateral OA, PF OA, Synovitis, Contusion.

## Live
- `m-c757b59738-r1` (pod tnygdbn5ph1645, $0.34/h, launched 10:13 IST, ~4.9 h): bestof_v1 k=8 v5 6 ep,
  **label_weights weak5x2.5**, seeds 42/1337 → members m2f53a8b148b, mee1b4cdbc94. (First launch at 10:08
  ran stale loop code without the flag; its pod vanished and the relaunch was verified to carry it.)
- `m-14225b510e` (strategist: relabel_0ed2240f Synovitis set, seeds 42/1337): pod died at 0.5 h before any
  arm uploaded; loop is polling results (retry n/8).
- Built, gate-checked, **not pushed** (need user OK): `outputs/kernels/tail-pf-k16` (swap soupbestof4 +
  PF OA←soupk16 w0.4) and `tail-pfsyn-k16` (PF OA + Synovitis ←soupk16 w0.35).

## Money
- Campaign ≈ $80 of $300. **RunPod $9.54 at 10:25 IST** (weak5 job ≈ $1.7). Top up ~$20 before Oct 3. DeepSeek ≈ $30.

## Submission calendar (≥1/day; gate ≤5/day; every push/submit needs user OK)
| day | candidate | depends on |
|---|---|---|
| Oct 1 | tail-pf-k16 (+ tail-pfsyn-k16 as A/B) | ready |
| Oct 2 | weak5x2.5 soup: `--tail "weak5\|soupw5\|0.3"` or swap-v5 soup(bestof4+w5) | weak5 members ≥ bestof k8 on the weak five |
| Oct 3 | Synovitis relabel members as Synovitis tail | m-14225b510e recovering |
| Oct 4–9 | one/day: combine LB winners; raise w only on tails that moved LB | — |
| Oct 10–14 | finals A (best public LB) / B (best expert-58 macro within LB −0.001) | — |
| Oct 16 | lock finals | — |
One leg checkpoint per kernel (soups, not raw members) until a 2-checkpoint leg is shown to fit 9 h.

## Next 3 actions
1. User OK → push/submit tail-pf-k16 (and A/B tail-pfsyn-k16); record LB in submits.json.
2. Collect weak5x2.5 members (~15:30 IST); per-finding vs bestof_v1 k=8; soup if weak five improve.
3. Optional: `members eval --targets soupk16` to confirm the soup keeps PF OA ≈ 0.88.
