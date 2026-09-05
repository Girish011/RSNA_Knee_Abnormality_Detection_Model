# Imaging Playbook Series (greenfield)

Public writeups live in [`site/`](../site/). Competition engineering memory stays in this `docs/` folder.

## Stance

- **From scratch:** plan and build a new pipeline. Do not resume the prior stack as the solution.
- **Prior probes:** kill ledger / design constraints only (what not to repeat). Not a live scoreboard.
- **Tone:** professional technical posts. No learning-path framing on the site.

## Method references

- [Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- [GenAI-assisted coding](https://developer.nvidia.com/blog/winning-a-kaggle-competition-with-generative-ai-assisted-coding/)

## Post roadmap

| Post | Topic |
|---|---|
| 01 | Playbook remapped + greenfield thesis + kill ledger |
| 02 | EDA and problem shape |
| 03 | Greenfield baselines |
| 04 | Efficiency-aware volume recipe |
| 05 | Frozen teacher / report distillation |
| 06 | Ensembles under 9h submit |
| 07 | Extra training after ruler wins |
| 08 | Dual finals (max-AUC + efficiency student) |

## Greenfield thesis (default)

1. Freeze the report teacher (no label micro-tuning wars).
2. Efficiency-aware volume + study encoder.
3. Single-model gold bar before ensembles.
4. Efficiency student in parallel.
5. Submit/decode path first-class.
6. Ruler: full-58 gold macro (margin 0.005); weak-val is smoke only.

## Drafting Post 01

Edit [`site/posts/01-imaging-playbook.md`](../site/posts/01-imaging-playbook.md), then mirror finished prose into [`site/posts/01-imaging-playbook.html`](../site/posts/01-imaging-playbook.html).
