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
5. **DeepSeek strategist** (V4.1-Flash) works every 45 min while pods run: relabels weak findings, keeps the member queue stocked, queues label/corpus fold-0 trains, builds (not pushes) kernels. ≤$15/day.

## Live
| Job | Pod | $/h | What |
|---|---|---:|---|
| `n-ca830ada7798` | done | 0.34 | bb_lr=1e-5 → proxy **0.9200**, gold_dev 0.8775 — miss; lr axis closed (~$1.02) |
| `m-f8709fd3f8` | `n6b0pdsjxzuzwq` | 0.74 | **v5 warm-start**, full data, 6 ep, labels `relabel_8c56cdf0` + `dread` (2 arms, est ≤$3.6). Launched 01:50 IST; downloading dense80. Monitor https://n6b0pdsjxzuzwq-8000.proxy.runpod.net |

**Submitted** `ens-d4-5bc0-w20` v1 (D4 + 5bc0 w=0.2) → ref **56651119** PENDING. Record: `scripts/submit_gate.py record --ref 56651119 --score <lb>`.

## Money
- Campaign: ~$34 of $300. RunPod balance **$10.06** (low). DeepSeek $47.94 (fine).

## Next 3 actions
1. When `m-f8709fd3f8` lands: compare member gold_dev/holdout vs v5 parent (0.921 gold-58) and 5bc0.
2. Build `ens-d4-<best members>` kernel; user approves push → run → `submit_gate.py submit --i-approve`.
3. Second member job: best recipe × 2–3 seeds + label diversity (weak findings: Synovitis, Lat OA, PF OA).
