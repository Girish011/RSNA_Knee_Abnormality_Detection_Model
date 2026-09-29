# STATUS

Last updated: 2026-09-29 (~01:15 IST)

## Where we really are (LB)
| What | Public LB |
|---|---:|
| Top of board (rhoskeri) | 0.960 (20 teams ≥ 0.955) |
| Public D4 stack (`jiweiliu/rsna-knee-fast-2xt4-inference`) | 0.943 |
| Public single CoAtNet, full 4349 corpus | 0.924 |
| **Ours** — single fold-0 dense80 CoAtNet `5bc0` | **0.920** (rank ~2352/4526) |

Gap is **ensemble strength**, not hyperparameters. lr/wd/head_lr/epoch micro-diffs all within ±0.003 proxy → axis closed.
Deadline **2026-10-22**.

## Strategy (DECISIONS 2026-09-29)
1. Submissions = **public D4 stack + our members, rank-blended** (`scripts/build_ensemble_kernel.py`).
2. Members = **full-data** retrains (no fold), several arms per pod (`python -m rsna_knee.agent.members …`).
3. **Warm start from public v5** full-SWA CoAtNet (CC0; 486/486 tensors load; public fine-tune → 0.932 standalone).
4. Every LB submit through `scripts/submit_gate.py` (approval flag, 5/day, duplicate refusal, ledger).
5. **DeepSeek strategist** (V4.1-Flash) works every 45 min while pods run: relabels weak findings, keeps the member queue stocked, queues label/corpus fold-0 trains, builds (not pushes) kernels. Budgets: relabels ≤$15/day, thinking ≤$3/day (separate).

## Live (2026-09-29 09:05 IST)
Full-data members (all v5 warm start, 6 ep) — holdout-29 vs 5bc0's 0.915:
| Member | Labels | seed | gold_dev | holdout |
|---|---|---:|---:|---:|
| m6edbb53fbab | relabel_8c56cdf0 | 42 | 0.910 | **0.939** |
| mb40244a6b78 | dread | 42 | 0.918 | 0.934 |
| m17d7af073bf | relabel_c7ab23d1 (DeepSeek Syn/PF/LatOA) | 0 | 0.917 | 0.934 |
| m3e9aed3d0c9 | relabel_c7ab23d1 | 1 | 0.917 | 0.936 |
| m1df0b4ac76d | relabel_c7ab23d1 | 7 | 0.914 | 0.927 |
| mdbe175a745e | relabel_d19bf15e (DeepSeek LatOA/PF) | 7 | 0.916 | 0.934 |

**Best label set: `bestof_v1`** — members m470c713b9fe / m0b16e9a09f1 macro-58 **0.932** (gold_dev 0.926 / 0.922, holdout 0.938 / **0.942**). 15 members done.
Running: fold-0 `n-6e0c83b59036` on `n6n5lchdmm4jiz` ($0.74/h; launched by the retired idle planner — user may terminate). Fold-0 trains now disabled.
LB: `ens-d4-5bc0-w20` ref 56651119 = **0.941** vs base D4 V11 **0.943** (Kaggle page) → a weak (0.920) tail leg at w=0.2 costs ~0.002.
Submitted **`swap-d4-m6ed`** → ref **56665556** PENDING (audit confirms our checkpoint loaded in both v5 arms). Also built, not pushed: (our best member replaces v5 in the dominant maxspan arms, 70% of Raptor family; no tail) and `ens-d4-m6-w30` (tail leg, 6 members).

## Money
- Campaign: ~$34 of $300. RunPod balance **$10.06** (low). DeepSeek $47.94 (fine).

## Next 3 actions
1. When `m-f8709fd3f8` lands: compare member gold_dev/holdout vs v5 parent (0.921 gold-58) and 5bc0.
2. Build `ens-d4-<best members>` kernel; user approves push → run → `submit_gate.py submit --i-approve`.
3. Second member job: best recipe × 2–3 seeds + label diversity (weak findings: Synovitis, Lat OA, PF OA).
