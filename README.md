# RSNA Knee Abnormality Detection

Competition: [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

Study-level multilabel MRI classifier wrapping **Meta DINOv2**, with report-derived weak labels for training only, dual finals for **main** and **efficiency** prizes.

## Hard rules
- Inference is **image + series metadata only** (no reports at test time).
- Repo docs are the memory across sessions — see `docs/STATUS.md`.
- Do not chase public LB without an OOF win.

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
src/rsna_knee/     # library code
configs/           # baseline / main / efficiency YAML
docs/              # STATUS, DECISIONS, experiments, weekly
notebooks/         # Kaggle audit / train / submit notebooks
tests/             # unit tests (no DICOM required)
```

## Continuity
Before any new chat session, read:
1. `docs/STATUS.md`
2. `docs/DECISIONS.md`
3. last rows of `docs/experiments.md`

## Blockers
- Complete Kaggle identity verification
- Attach competition data on Kaggle (569 GB does not fit on this Mac — ~129 GB free)
- Bundle DINOv2 weights as a public Kaggle Model/Dataset for offline submit
