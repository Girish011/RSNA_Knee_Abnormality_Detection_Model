# STATUS

Last updated: 2026-10-01 (~21:00 IST). See docs/HANDOFF.md for the full picture.

## Leaderboard
| ref | kernel | public |
|---|---|---:|
| 56651119 | D4 + 5bc0 tail w=0.2 | 0.941 |
| 56665556 | swap m6edbb53fbab → v5 | 0.943 |
| 56680743 | swap m0b16e9a09f1 → v5 | 0.943 |
| **56704765** | **swap soupbestof4 → v5** | **0.944 (rank 258/4660)** |
| 56744344 | + PF OA ← soupk16 w0.4 | 0.944 |
| 56744346 | + PF OA, Synovitis ← soupk16 w0.35 | 0.944 |

## Best models (AUC on 58 experts)
- soupbestof4 0.9365 (in LB 0.944); bestof_v1 k=8 members 0.9330; public v5 0.9256.
- k=16 bestof members 0.9266 overall but **PF OA 0.882** (vs 0.855) → `soupk16` (2-seed soup, uploaded
  `girishbose/rsna-knee-soup-k16`, not yet expert-evaluated) = PF OA specialist leg.
- Weak findings everywhere: Lateral Meniscus, Lateral OA, PF OA, Synovitis, Contusion.

## Live
- No pods. Queued (unapproved, strategist): `m-d8cace1ba0` 3 arms from node 6e0c83b59036, ~$12.28 — recommend drop.
- **weak5x2.5 members (macro-58, mean of 2):** macro 0.9321 (bestof 0.9330); weak five 0.8867 (0.8858);
  **Synovitis 0.888 (best ever; bestof 0.863)**, PF OA 0.862, Contusion 0.921 (−0.022). → Synovitis specialist
  `soupw5`. `relabel_d19bf15e` (6 seeds, strategist bypassed the seed cap — fixed) macro 0.929, PF OA 0.837: no.
- Built, gate-checked, **not pushed**: `tail-syn-w5` (swap soupbestof4 + Synovitis ← soupw5 w0.4) and
  `swap-d4-b4w5` (6-member soup bestof4 + weak5x2.5 in the v5 slot; unevaluated).

## Money
- Campaign ≈ $85 of $300. **RunPod $4.10 at 20:30 IST**; user tops up 2026-10-02. DeepSeek ≈ $30.

## Submission calendar (≥1/day; gate ≤5/day; every push/submit needs user OK)
| day | candidate | depends on |
|---|---|---|
| Oct 1 | tail-pf-k16 (+ tail-pfsyn-k16 as A/B) | ready |
| Oct 2 | tail-syn-w5 and/or swap-d4-b4w5 | built |
| Oct 3 | Synovitis relabel members as Synovitis tail | m-14225b510e recovering |
| Oct 4–9 | one/day: combine LB winners; raise w only on tails that moved LB | — |
| Oct 10–14 | finals A (best public LB) / B (best expert-58 macro within LB −0.001) | — |
| Oct 16 | lock finals | — |
One leg checkpoint per kernel (soups, not raw members) until a 2-checkpoint leg is shown to fit 9 h.

## Next 3 actions
1. User OK → submit tail-syn-w5 / swap-d4-b4w5 (Oct 2).
2. Single-finding tails do not show at 3 decimals (both Oct 1 A/Bs = 0.944): next GPU money should buy a
   bigger macro move (e.g. a weak-five recipe that keeps Contusion: lower weight on Contusion), not more tails.
3. Drop `m-d8cace1ba0` unless the user wants it.
