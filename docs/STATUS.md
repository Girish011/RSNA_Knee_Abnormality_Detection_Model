# STATUS

Last updated: 2026-10-01 (~22:30 IST). See docs/HANDOFF.md for the full picture.

## Leaderboard
| ref | kernel | public |
|---|---|---:|
| 56651119 | D4 + 5bc0 tail w=0.2 | 0.941 |
| 56665556 | swap m6edbb53fbab → v5 | 0.943 |
| 56680743 | swap m0b16e9a09f1 → v5 | 0.943 |
| **56704765** | **swap soupbestof4 → v5** | **0.944 (rank 258/4660)** |
| 56744344 | + PF OA ← soupk16 w0.4 | 0.944 |
| 56744346 | + PF OA, Synovitis ← soupk16 w0.35 | 0.944 |
| 56754652 | LEARNING raptor-all-b4: soup4 in v5+v8+v10 | pending |
| 56754655 | LEARNING coat-up15-b4: soup4 swap + CoAt weight +0.15 | pending |

## Best models (AUC on 58 experts)
- soupbestof4 0.9365 (in LB 0.944); bestof_v1 k=8 members 0.9330; public v5 0.9256.
- k=16 bestof members 0.9266 overall but **PF OA 0.882** (vs 0.855) → `soupk16` (2-seed soup, uploaded
  `girishbose/rsna-knee-soup-k16`, not yet expert-evaluated) = PF OA specialist leg.
- Weak findings everywhere: Lateral Meniscus, Lateral OA, PF OA, Synovitis, Contusion.

## Live
- **Label transplant (Kaggle GPU, free):** `girishbose/fork-e13-bestof-v1` (E13 RadImageNet heads on bestof_v1; upstream
  gold OOF 0.8576) and `girishbose/fork-e9main-bestof-v1` (main V52 heads; upstream 0.8543). Read `rad_e11_audit.json` /
  `rad_e9_audit.json` (gold_oof / rad_gold_macro) when they finish; win → wire heads into the stack (patch SHA + path).
- No RunPod pods. Strategist is propose-only. weak5x2.5 = Synovitis specialist (`soupw5`); kernels tail-syn-w5 /
  swap-d4-b4w5 built but NOT to be submitted (expected 0.944).

## Money
- Campaign ≈ $85 of $300. **RunPod $4.10 at 20:30 IST**; user tops up 2026-10-02. DeepSeek ≈ $30.

## Submission policy
Only submit candidates expected to beat 0.944 (or explicit learning probes). Gate ≤5/day.

## Next 3 actions
1. Read the two learning probes (raptor-all-b4, coat-up15-b4) → decides whether more CoAt (v8/v10 fine-tunes) pays.
2. Read the two fork audits; wire winning heads into the stack (cell 25: `_RAD_HEADS_SHA256`, `_RAD_E13_HEADS_SHA256`,
   candidate paths; check the calibrator is not fit to the public heads) → candidate submit.
3. Next transplant targets: A5 folds / DINO members (find public trainers), then CoAt readers.
