# Higher path: 0.943 → ≥ 0.955 (scoped 2026-09-29)

## Why the current path plateaus
Public D4 final blend = transformer/RadImageNet branch + CoAt/Raptor branch (CoAt weight 0.60; 0.75–1.0 for ACL,
Med Men, Lat Men, Lat OA, Fracture). CoAt branch = 4 families at 0.25 each; v5 = 70% of the Raptor family.
Our swap touches ≈ 0.6 × 0.25 × 0.7 ≈ **10% of the final prediction** → a member +0.015 better on experts moves the
LB ≈ +0.0015. Swaps/soups cap out around 0.945.

## Path (each phase gated on the previous phase's measurement)
| Phase | What | Infra | GPU $ | Other $ | Days | Gate to continue |
|---|---|---|---:|---:|---:|---|
| 0 | **Expert bench**: run the whole public stack on the 58 expert TRAIN studies (Kaggle kernel, test root → expert DICOMs) → every family's predictions on ground truth. Eval-only jobs for v5 / soups (built). | Kaggle GPU (free quota) | 0–2 | 0 | 1–2 | Know per-family / per-finding AUC; where ours beats public |
| 1 | **Our family as a 5th CoAt reader** with bench-set weight; retrain members at **k=16 on 48 GB** (A40 / RTX A6000 $0.33–0.35/h lifts the 24 GB k=8 ban); swap more slots | RunPod 48 GB class | 10–20 | 0 | 3–5 | LB ≥ 0.946 |
| 2 | **New strong family** for diversity: different backbone (e.g. MaxViT-B / ConvNeXt-B 384 or DINOv2-B fine-tune) on dense80 + best-of labels, 4–6 full-data models; **full 12-finding relabel** with per-finding best-of judged by members | A40/A6000 (A100 80 GB only if 48 GB OOMs, $1.19/h) | 30–60 | DeepSeek 20–30 | 7–10 | LB ≥ 0.950 |
| 3 | Final blend tuned on the expert bench (shrunk toward public weights), runtime check vs 9 h, **dual finals** (max-AUC + safe) | Kaggle | 0 | 0 | 2 | — |

## Totals (to 22 Oct)
- RunPod: **+$60–90** beyond today's balance (~$6); campaign stays ≈ $140–180 of the $300 cap (spent $63).
- DeepSeek: **+$20–30** (balance ~$40).
- Kaggle: free GPU quota (bench + submit kernels); watch the weekly 30 h.
- Honest odds by 22 Oct: 0.947–0.950 ~50%; 0.955 ~25%; 0.958+ ~10% (vs ~10% / ~3–5% on the swap-only path).

## Risks
- Expert bench leakage: some public families may have trained on the 58 experts → their bench AUC is optimistic; use it for relative per-finding decisions, shrink weights toward public.
- 58 experts cannot support 12 free weights: one global weight per branch plus at most a few per-finding overrides.
- Runtime: every added family costs Kaggle minutes; the stack + our leg must stay < 9 h on the hidden set.
- License: public checkpoints used in a final must be open (v5 dataset is CC0; check each new one).
