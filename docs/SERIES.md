# Imaging Playbook Series (greenfield)

Public writeups live in [`site/`](../site/). Competition engineering memory stays in this `docs/` folder.

Full locked plan (recovered from the original planning chat): [`docs/rsna_playbook_blog.plan.md`](rsna_playbook_blog.plan.md).

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
| 01 | Playbook remapped + starting plan + what already failed |
| 02 | Looking at the data carefully (EDA: exploratory data analysis) |
| 03 | First simple models from scratch |
| 04 | Smarter sampling of the MRI exam |
| 05 | Fair text teacher / labels from reports (rule 2.6.b) |
| 06 | Combining models under the 9-hour limit |
| 07 | Extra training after the yardstick says yes |
| 08 | Best-AUC final + efficiency student |

## Writing style (locked)

- Explain acronyms on first use (MRI, AUC, LLM, OOF, and so on).
- Prefer plain English a non-specialist can follow; keep facts precise.
- No em dashes.
- Professional contest writeup tone (not a “learning path” voice).

## Greenfield thesis (default)

1. Lock a rules-compliant teacher (accessible / minimal-cost LLM or public model OK for **train-time** report labels under §2.6.b; never in submit).
2. Efficiency-aware volume + study encoder.
3. Single-model gold bar before ensembles.
4. Efficiency student in parallel.
5. Submit/decode path first-class (images + series metadata only).
6. Ruler: full-58 gold macro (margin 0.005); weak-val is smoke only.

## Host rules we must keep in view

- Commercially hosted LLMs allowed for train-time processing (e.g. label extraction from reports) if reasonably accessible and minimal cost ([rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) §2.6.b).
- That is not private sharing by itself; private sharing still bars out-of-team collaboration.
- Host may disallow unfair / prohibitively costly setups.
- **KneeCoT is not allowed** (gated HF access + institutional agreement → uneven field). Do not use for auxiliary training or distillation unless the host reverses stance.
- Submit notebook: offline, no reports.
- Prefer external assets that are public, offline-bundleable for submit, and license-documented without one-off institutional approvals.

## Drafting

- Post 01: filled in [`site/posts/01-imaging-playbook.md`](../site/posts/01-imaging-playbook.md)
- Post 02: filled in [`site/posts/02-eda-problem-shape.md`](../site/posts/02-eda-problem-shape.md) (from Step A audit + plots)
