# RSNA Knee Abnormality Detection

Competition: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

Public monorepo for a **greenfield** campaign: code + technical writeup series adapting the Kaggle Grandmasters Playbook to multimodal knee MRI (code competition, dual finals for **main** and **efficiency**).

## Series / site

- Writeups: [`site/`](site/) (GitHub Pages source)
- Series plan: [`docs/SERIES.md`](docs/SERIES.md)
- Post 01 outline: [`site/posts/01-imaging-playbook.md`](site/posts/01-imaging-playbook.md)
- Pages URL (after first deploy): `https://girish011.github.io/RSNA_Knee_Abnormality_Detection_Model/`

Prior experiment numbers are a **kill ledger** for redesign — not a continuing scoreboard.

## Hard rules
- Inference is **image + series metadata only** (no reports at test time).
- Repo docs are the memory across sessions - see `docs/STATUS.md` and `docs/SERIES.md`.
- Do not chase public LB without an OOF win on the trusted ruler (full-58 gold macro).

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Torch / DINOv2 training (Kaggle or machine with GPU):
```bash
pip install -e ".[torch]"
```

## Layout
```text
site/              # public blog (GitHub Pages)
src/rsna_knee/     # library code
configs/           # baseline / main / efficiency YAML
docs/              # STATUS, SERIES, DECISIONS, experiments
notebooks/         # Kaggle audit / train / submit notebooks
tests/             # unit tests (no DICOM required)
```

## Continuity
Before any new chat session, read:
1. `docs/SERIES.md`
2. `docs/STATUS.md`
3. `docs/DECISIONS.md`
4. last rows of `docs/experiments.md`

## Blockers
- Complete Kaggle identity verification
- Attach competition data on Kaggle (569 GB does not fit on this Mac - ~129 GB free)
- Bundle DINOv2 weights as a public Kaggle Model/Dataset for offline submit
