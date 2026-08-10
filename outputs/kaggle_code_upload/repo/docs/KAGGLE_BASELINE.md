# Kaggle baseline runbook (cache → DINOv2-S)

Do this on Kaggle. Keep the 569 GB DICOMs there.

## 0. Upload code as a Dataset
From this Mac (after latest push), either:
- Zip the repo (without `.venv`) and **New Dataset** → name: `rsna-knee-code`
- Or keep syncing via GitHub and download zip on Kaggle with internet (private repo needs token)

Include at least: `src/`, `scripts/`, `configs/`, `data/folds/folds_v1.csv`, optionally `data/processed/weak_labels_v1.csv`.

## 1. DINOv2-S weights (public)
On Mac:
```bash
source .venv/bin/activate
python scripts/fetch_dinov2_weights.py
```
Upload `data/external/dinov2/` as public Kaggle Dataset: `dinov2-vits14-rsna-knee`  
(or attach the existing Kaggle Model Hub **DINOv2 small** if paths match).

## 2. Build cache
1. Competition notebook + attach: competition data, `rsna-knee-code`
2. Upload / run [`notebooks/02_build_cache.ipynb`](../notebooks/02_build_cache.ipynb)
3. Smoke: `LIMIT=50` → then `LIMIT=0` full
4. Save Version → **New Dataset** `rsna-knee-cache-v1` from output `cache_v1/`

Cache recipe: max 3 series × 12 slices × 224 uint8 (~few GB).

## 3. Train baseline fold
1. New GPU notebook
2. Attach: `rsna-knee-cache-v1`, `rsna-knee-code`, `dinov2-vits14-rsna-knee` (+ competition data if train.csv not in code dataset)
3. Run [`notebooks/10_train_baseline.ipynb`](../notebooks/10_train_baseline.ipynb)
4. Start fold 0 / 3 epochs smoke → then 5 folds × 5 epochs
5. Download metrics/checkpoints; log in `docs/experiments.md`

## 4. First submit (after OOF looks sane)
Use `notebooks/90_submit_main.ipynb` with internet **off**, weights + code attached, write `submission.csv`.

## Local Mac role
- Improve code / weak labels / folds
- `git push`
- Re-upload code dataset when scripts change
- Never download full `train_series/`
