#!/usr/bin/env bash
# Manual recovery entrypoint; normal launch embeds the supervisor so boot is bounded.
set -euo pipefail
exec python -u /workspace/bundle/definition_job.py
