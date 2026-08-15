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
ROOT="${ROOT}" python3 - <<'PY'
import os
from pathlib import Path
import pandas as pd

root = Path(os.environ["ROOT"])
dst = root / "outputs/kaggle_code_upload/repo/data/processed"
dst.mkdir(parents=True, exist_ok=True)
for name in ("weak_labels_v1.csv", "weak_labels_v2.csv", "weak_labels_v3.csv"):
    src = root / "data/processed" / name
    if not src.exists():
        print("skip missing", name)
        continue
    df = pd.read_csv(src)
    cols = [c for c in df.columns if c != "Report"]
    out = dst / name
    df[cols].to_csv(out, index=False)
    print("packed", name, len(df), "cols", len(cols))
PY

# Stage dinov2 weights for separate dataset uploads
mkdir -p "${OUT}/dinov2_s" "${OUT}/dinov2_b"
if [[ -f "${ROOT}/data/external/dinov2/dinov2_vits14_pretrain.pth" ]]; then
  cp "${ROOT}/data/external/dinov2/dinov2_vits14_pretrain.pth" "${OUT}/dinov2_s/"
fi
if [[ -f "${ROOT}/data/external/dinov2/dinov2_vitb14_pretrain.pth" ]]; then
  cp "${ROOT}/data/external/dinov2/dinov2_vitb14_pretrain.pth" "${OUT}/dinov2_b/"
fi

(
  cd "${OUT}"
  rm -f rsna-knee-code.zip dinov2-vits14-rsna-knee.zip dinov2-vitb14-rsna-knee.zip
  (cd repo && zip -qr ../rsna-knee-code.zip .)
  if [[ -f dinov2_s/dinov2_vits14_pretrain.pth ]]; then
    (cd dinov2_s && zip -qr ../dinov2-vits14-rsna-knee.zip .)
  fi
  if [[ -f dinov2_b/dinov2_vitb14_pretrain.pth ]]; then
    (cd dinov2_b && zip -qr ../dinov2-vitb14-rsna-knee.zip .)
  fi
)

ls -lh "${OUT}"/*.zip
echo "Upload zips at https://www.kaggle.com/datasets → New Dataset / New Version"
echo "  rsna-knee-code.zip  (update private code dataset)"
echo "  dinov2-vits14-rsna-knee.zip  (public)"
echo "  dinov2-vitb14-rsna-knee.zip  (public — required for B runs)"
