# An Imaging Playbook for RSNA Knee Abnormality Detection

> Status: **outline stub** — fill each section. Do not paste prior LB as “current standing.”
> Tone: professional technical writeup (Grandmaster-style). No learning-path / career framing.

## 1. Series framing

TODO: Grandmasters Playbook + agent-assisted loop applied to this multimodal code competition.
Cite:
- https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
- https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/

## 2. What the problem actually is

TODO: study-level 12-label macro AUC; images + series metadata at test; reports train-only; scale; ≤9h offline submit; dual finals.

## 3. The two exams problem

TODO: noisy report teachers vs expert-graded images; measurement trap table (rulers, not live standing).

### Multilingual teacher (brief)

TODO: FR / TR (~600) / ES / DE / EL (~320) / NL / EN; one negation example. Deep recipes → Post 5.

## 4. Foundations remapped

TODO: fast experimentation under GPU quota; full-58 gold macro keep/kill; weak-val is smoke only.

## 5. Four-step agent workflow

TODO: EDA → baselines → improve → combine; agents write code; rulers decide what ships; save OOF/preds.

## 6. Constraints from prior probes (kill ledger)

TODO: short table of what already failed — as design constraints for a greenfield plan, not a continuing scoreboard.

## 7. Greenfield thesis

TODO:
1. Freeze the teacher (no label micro-tuning wars).
2. Efficiency-aware volume + study encoder.
3. Single-model gold bar before ensembles.
4. Efficiency student in parallel.
5. Submit/decode path first-class.

## 8. Series roadmap

TODO: map Posts 2–8 to remapped playbook techniques.

## 9. Attribution

TODO: NVIDIA posts; competition page; disclaimer that prior probe numbers are historical constraints.
