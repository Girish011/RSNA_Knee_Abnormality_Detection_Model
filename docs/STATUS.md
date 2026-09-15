# STATUS

Last updated: 2026-09-15

## Phase
**Greenfield Imaging Playbook**. Labels/teacher track.
Coding: **gf_labels_lig2 5-fold OOF launched** (Med-Men-only gap-fill). Floor remains gf_v0 **0.6144**.

## Done
- Posts 01–03; volume v1/v2 KILL; gf_v0 true OOF gold **0.6144**
- gf_labels_lig1 (gap-fill ACL/MCL/Med Men on weak_v1) → OOF gold **0.6019** → **KILL**
- Built `weak_labels_gf_lig2.csv`: weak_v1 + gap-fill **Medial Meniscus only** (drop MCL and ACL fills)

## Active experiment
- Recipe: same `cache_gf_v0` / frozen DINOv2-S / seed 42 / 5 folds
- Teacher: `data/processed/weak_labels_gf_lig2.csv` (Med Men 866→1460; ACL/MCL = weak_v1)
- Kernel: `girishbose/gf-labels-lig2-5fold` (T4)
- Meta: `girishbose/rsna-knee-gf-lig2-meta`
- Keep if true OOF gold ≥ **0.6194**
- Watch: Med Men gold OOF (lig1 was +0.128) and whether Effusion/Lat OA still collapse

## Scoreboard
| Run | Weak OOF | Gold OOF | Verdict |
|---|---|---|---|
| gf_v0 weak_v1 | 0.685 | **0.6144** | floor / KEEP |
| gf_labels_lig1 | 0.707 | **0.6019** | KILL (Δ −0.0125) |
| gf_labels_lig2 | — | — | running |

## Next 3 actions
1. User reports when `gf-labels-lig2-5fold` finishes (do not poll).
2. Pull OOF gold + per-label (esp. Med Men, Effusion); KEEP/KILL vs **0.6194**.
3. If KILL: next candidate is high-precision Med Men fills (raise min_conf) or Med Men+tight ACL — one axis only.

## Do not
- Poll Kaggle; adopt lig1 as default; reports at test time; KneeCoT
- Treat weak OOF lift as a win; treat fold0→all58 **0.728** as the floor
