# Kaggle workflow (569 GB stays in the cloud)

Local disk has ~129 GB free. **Do not download `train_series/` or `test_series/`.**

## A. One-time setup (API token)

1. Open https://www.kaggle.com/settings/api → **Generate New Token**
2. Preferred (modern):

```bash
mkdir -p ~/.kaggle
# paste the KGAT_… token into the file (one line, no quotes)
nano ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Or legacy `kaggle.json` via **Create Legacy API Key** → `~/.kaggle/kaggle.json`.

3. On this Mac:

```bash
cd /Users/girish11/RSNA_Knee_Abnormality_Detection_Model
source .venv/bin/activate
pip install -U 'kaggle>=1.7'
kaggle competitions list -s knee   # smoke-test auth
```

**Security:** never paste tokens into chat, commits, or screenshots. If a token is exposed, revoke it on the settings page and generate a new one.

## B. Local: metadata only (~few MB)

```bash
./scripts/download_metadata.sh
```

This pulls only:
- `train.csv`, `train_series.csv`
- `test.csv`, `test_series.csv`
- `sample_submission.csv`

Then we can freeze folds + prototype weak labels **without any DICOMs**.

## C. Kaggle: full DICOM audit / cache / train

1. Go to the competition → **Code** → **New Notebook**
2. Settings:
   - Accelerator: **GPU T4** (or P100) for train; CPU OK for CSV audit
   - Internet: **On** for install/debug; **Off** when submitting
   - Persistence: Files only (optional)
3. Add data: competition data is auto-attached for competition notebooks
4. File → Import the repo notebook, **or** paste cells from:
   - `notebooks/01_data_audit.ipynb`
5. Run all → **Save Version** → download the output tables into `docs/audit/`

### Recommended notebook sequence on Kaggle
| Notebook | Purpose | Needs DICOMs? |
|---|---|---|
| `01_data_audit` | prevalence, languages, series stats | No (CSV) + optional DICOM spot-check |
| `02_build_cache` | resized slice cache → publish as Dataset | Yes |
| `10_train_baseline` | DINOv2-S train | Cache preferred |
| `90_submit_main` | offline `submission.csv` | Yes (test) |

## D. What never lives on the Mac
- Full `train_series/` (~hundreds of GB)
- Full model training on MPS for the whole dataset (prototype only)

## E. Session handoff
After metadata download, update `docs/STATUS.md` and continue with folds + weak-label audit locally while Kaggle runs the heavy jobs.
