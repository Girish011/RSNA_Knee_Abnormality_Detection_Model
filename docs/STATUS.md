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

## LB (2026-09-29 21:20 IST)
| ref | kernel | public |
|---|---|---:|
| 56651119 | D4 + 5bc0 tail w=0.2 | 0.941 |
| 56665556 | swap m6edbb53fbab → v5 arms | **0.943** (= base; tail penalty gone) |
| 56680743 | swap m0b16e9a09f1 (bestof_v1) → v5 arms | PENDING |
Forward/reverse maxspan arms share ONE model → several members only as a weight soup. `soupbestof2` (m0b16 + m470c) built; backbones 1% apart, heads 40–70% apart → validate before any LB use. Eval job `e-dd366d455d` (v5 + soupbestof2 on our dev/holdout) queued after `m-6c83c7d7f7`.

## Now (2026-09-30 00:30 IST) — self-training round 1
- **2 pods allowed** (user). Running: `f-7cdb6e8357` 5-fold OOF (bestof_v1, v5, 6 ep) on `t5ibizzt71f4l5` (A40 $0.49/h); `m-6c83c7d7f7` (bestof seeds) on slow 3090 `6xc11q3uqpdrdi`.
- Queue: `e-dd366d455d` (v5 + soupbestof2 eval) → `m-c15cb7b0ea`.
- Next: `scripts/fuse_labels.py --oof-job f-7cdb6e8357 --write fused_v1` → members on fused_v1 → per-finding routing.
- Launcher order (deadline mode): community 4090 → 48 GB A6000/A40 → 3090 → secure.

## Money
- Campaign: ~$34 of $300. RunPod ~$31 after $25 top-up. DeepSeek $47.94 (fine).

## Next 3 actions
1. When `m-f8709fd3f8` lands: compare member gold_dev/holdout vs v5 parent (0.921 gold-58) and 5bc0.
2. Build `ens-d4-<best members>` kernel; user approves push → run → `submit_gate.py submit --i-approve`.
3. Second member job: best recipe × 2–3 seeds + label diversity (weak findings: Synovitis, Lat OA, PF OA).
