#!/usr/bin/env bash
# Package code (+ folds + weak labels if present) for Kaggle Dataset upload.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${ROOT}/outputs/kaggle_code_upload"
rm -rf "${OUT}"
mkdir -p "${OUT}"

rsync -a \
  --exclude '.venv' \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude 'data/raw' \
  --exclude 'data/cache' \
  --exclude 'data/external' \
  --exclude 'outputs' \
  "${ROOT}/" "${OUT}/repo/"

mkdir -p "${OUT}/repo/data/processed"
if [[ -f "${ROOT}/data/processed/weak_labels_v1.csv" ]]; then
  ROOT="${ROOT}" python - <<'PY'
import os
import pandas as pd
from pathlib import Path
root = Path(os.environ["ROOT"])
df = pd.read_csv(root / "data/processed/weak_labels_v1.csv")
cols = [c for c in df.columns if c != "Report"]
out = root / "outputs/kaggle_code_upload/repo/data/processed/weak_labels_v1.csv"
out.parent.mkdir(parents=True, exist_ok=True)
df[cols].to_csv(out, index=False)
print("packed weak labels", len(df), "cols", len(cols))
PY
fi

# Also stage dinov2 weights alongside for separate dataset upload convenience
if [[ -f "${ROOT}/data/external/dinov2/dinov2_vits14_pretrain.pth" ]]; then
  mkdir -p "${OUT}/dinov2"
  cp "${ROOT}/data/external/dinov2/dinov2_vits14_pretrain.pth" "${OUT}/dinov2/"
fi

(
  cd "${OUT}"
  rm -f rsna-knee-code.zip dinov2-vits14-rsna-knee.zip
  (cd repo && zip -qr ../rsna-knee-code.zip .)
  if [[ -d dinov2 ]]; then
    (cd dinov2 && zip -qr ../dinov2-vits14-rsna-knee.zip .)
  fi
)

ls -lh "${OUT}"/*.zip
echo "Upload zips at https://www.kaggle.com/datasets → New Dataset"
echo "  rsna-knee-code.zip"
echo "  dinov2-vits14-rsna-knee.zip"
