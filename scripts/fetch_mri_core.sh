#!/usr/bin/env bash
# Fetch public Apache-2.0 MRI-CORE source for offline Kaggle packaging.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/third_party/mri_foundation"
if [[ -f "${DEST}/models/sam/build_sam.py" ]]; then
  echo "MRI-CORE source already present at ${DEST}"
  exit 0
fi

git clone --depth 1 https://github.com/mazurowski-lab/mri_foundation.git "${DEST}"
rm -rf "${DEST}/.git"
if [[ ! -f "${DEST}/LICENSE" ]]; then
  # Upstream README declares Apache-2.0 but its LICENSE link is currently absent.
  curl -L --fail --output "${DEST}/LICENSE" \
    "https://www.apache.org/licenses/LICENSE-2.0.txt"
fi
echo "Vendored MRI-CORE source and Apache-2.0 LICENSE at ${DEST}"
echo "Download MRI_CORE_vitb.pth from the official repository link into data/external/mri_core/"
