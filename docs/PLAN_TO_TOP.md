# Plan to the top (2026-10-01 → lock 2026-10-16)

Target: public LB 0.950+ (top-86 band) on the way to the top band (0.955–0.961). Current 0.944.

## The insight this plan is built on (measured 2026-10-01, 58 experts, pooled)
| Model family | Macro | Weak five | Lat Men | PF OA | Synovitis |
|---|---:|---:|---:|---:|---:|
| Public DINO base (OOF) | 0.843 | 0.800 | 0.67 | 0.80 | 0.78 |
| Public RadImageNet (OOF) | 0.811 | 0.789 | 0.69 | 0.81 | 0.76 |
| **Our CoAtNet (held-out)** | **0.882** | **0.867** | **0.88** | **0.87** | 0.81 |

Our CoAtNet line is the strongest family, and our fused labels agree with experts at **0.946** (report labels 0.883).
The public stack gives ~40% of its weight to the two weakest families, and every public model learned from report labels.

## Strategy: three moves, then one grand soup
1. **Cross-architecture distillation ("bring our intelligence into every model").** Retrain the weak public families
   on our fused labels with their own public trainers (`scripts/fork_public_trainer.py`, free Kaggle GPU). Fused labels
   were flat for CoAt (self-distillation) but are new information for DINO / RadImageNet, which never saw it.
   Order: RadImageNet E13 + main heads (running) → A5 folds → DINO members (largest share, plan quota).
2. **Rebalance the stack toward the strongest family.** We now hold OOF for DINO, RadImageNet and our 5-fold CoAt on
   all 4,407 train studies → fit per-finding branch weights on 4,407 studies (targets: fused labels; validated on the
   58 experts), instead of the hand-tuned public weights. Probe `coat-up15-b4` measures the direction tonight.
3. **Strengthen the CoAt family itself.** Our soups in every Raptor slot (probe `raptor-all-b4`), then v8/v10
   fine-tunes on fused/bestof labels (RunPod, after top-up); specialists (k16 PF OA, weak5 Synovitis) folded in.
4. **Grand soup / ensemble selection.** Every upgraded component + learned weights; greedy forward selection on OOF,
   confirmed on experts and LB. Two finals: best LB, best OOF+expert.

## Calendar (one LB submit per day minimum; only candidates expected to beat the current best)
| Date | Work | LB candidate |
|---|---|---|
| Oct 1 | probes raptor-all / coat-up15; 4 RadImageNet forks (bestof + fused) | probes |
| Oct 2 | read forks; wire winning heads (`--rad-heads`); OOF branch-weight fitter | heads swap |
| Oct 3 | learned branch weights (+ heads) | rebalanced stack |
| Oct 4–6 | A5 transplant (fused); v8/v10 fine-tunes if probe positive | best combined |
| Oct 7–11 | DINO members transplant (fused), quota-planned | best combined |
| Oct 12–14 | grand soup + ensemble selection | finals candidates |
| Oct 15–16 | lock finals A (best LB) / B (best OOF+expert) | — |

## Gates (each step is kept only if it measures up)
- A fork wins when its trainer's gold OOF macro (58) beats the upstream audit (E13 0.8576, main 0.8543).
- A stack change ships to LB when expert/OOF evidence says it beats the current best.

## Roles
- Claude: architecture, approvals, submissions (user-approved policy: only candidates expected to beat best).
- DeepSeek strategist: analysis + proposals only (GPU spend via host approval).
- Kaggle GPU (free, 2 concurrent, 30 h/week): forks. RunPod (top-up): CoAt fine-tunes.

## Results log
- 2026-10-01 22:30 — RadImageNet forks on bestof_v1 (gold OOF macro, 58): **E13 0.8612 vs upstream 0.8576 (+0.004;
  blend α0.5 0.8701 vs 0.8654)**; main V52 0.8388 vs 0.8543 (−0.016: MCL/MedMen/Contusion/Synovitis down). Mixed →
  not wired yet. Fused-label forks (expert agreement 0.946) running: fork-e9main-fused-full, fork-e13-fused-full.
  Heads/OOF saved under outputs/rad_heads/ (gitignored weights).
- 2026-10-02 04:15 — **LB coat-up15-b4 = 0.945 (new best)**; raptor-all-b4 = 0.944 (more soup in v8/v10 slots: no).
  **Fused-label forks win:** E13 heads gold OOF 0.8745 vs upstream 0.8576 (+0.017; bestof 0.8612); main heads 0.8594 vs
  0.8543 (+0.005; LatOA +0.05, Syn +0.035, MedMen +0.036, Fracture +0.06; LatMen −0.06, PF −0.04 — LatMen is CoAt-only
  in the stack). Cross-architecture distillation confirmed → A5 / DINO next on fused labels.
  Heads: `girishbose/rsna-knee-rad-heads-fused`. Candidate `coat15-radfused` (coat +0.15 + both fused heads) pushed.
