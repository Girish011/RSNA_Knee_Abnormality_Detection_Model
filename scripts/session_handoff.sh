#!/usr/bin/env bash
# Paste into a new chat to defeat context rot.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Competition: RSNA Knee Abnormality Detection"
echo "Read first:"
echo "  $ROOT/docs/STATUS.md"
echo "  $ROOT/docs/DECISIONS.md"
echo "  $ROOT/docs/experiments.md (last 10)"
echo "Rules: $ROOT/.cursor/rules/rsna-knee.mdc"
echo "Do not: retune folds casually, chase public LB without OOF win, use reports at inference"
tail -n 40 "$ROOT/docs/STATUS.md"
