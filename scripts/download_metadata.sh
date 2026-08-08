#!/usr/bin/env bash
# Download competition CSV metadata only (NOT the 569GB DICOM tree).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/data/raw"
mkdir -p "${OUT}"

# Activate project venv if present (needed for modern kaggle CLI).
if [[ -f "${ROOT}/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi

if ! command -v kaggle >/dev/null 2>&1; then
  pip install "kaggle>=1.7"
fi

# Modern auth: ~/.kaggle/access_token  OR  $KAGGLE_API_TOKEN
# Legacy auth: ~/.kaggle/kaggle.json
has_auth=0
if [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
  has_auth=1
elif [[ -f "${HOME}/.kaggle/access_token" ]]; then
  has_auth=1
elif [[ -f "${HOME}/.kaggle/kaggle.json" ]]; then
  has_auth=1
fi

if [[ "${has_auth}" -ne 1 ]]; then
  echo "No Kaggle credentials found."
  echo "Preferred: save API token to ~/.kaggle/access_token (chmod 600)"
  echo "  https://www.kaggle.com/settings/api  → Generate New Token"
  echo "Legacy: ~/.kaggle/kaggle.json from 'Create Legacy API Key'"
  exit 1
fi

SLUG="rsna-knee-abnormality-detection"
if ! kaggle competitions files -c "${SLUG}" >/dev/null 2>&1; then
  SLUG="rsna-knee-abnormalities-detection"
  echo "Using slug: ${SLUG}"
fi

FILES=(
  train.csv
  train_series.csv
  test.csv
  test_series.csv
  sample_submission.csv
)

echo "Downloading metadata CSVs into ${OUT} (competition: ${SLUG})"
for f in "${FILES[@]}"; do
  echo "→ ${f}"
  kaggle competitions download -c "${SLUG}" -f "${f}" -p "${OUT}" --force
  if [[ -f "${OUT}/${f}.zip" ]]; then
    unzip -o "${OUT}/${f}.zip" -d "${OUT}"
    rm -f "${OUT}/${f}.zip"
  fi
done

echo
echo "Done. Local metadata:"
ls -lh "${OUT}"/*.csv 2>/dev/null || ls -lh "${OUT}"
echo
echo "Next: python scripts/freeze_folds.py"
