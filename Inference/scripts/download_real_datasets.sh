#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

python -m src.download_datasets \
  --dataset all \
  --output-dir data \
  --activity-limit 1000 \
  --pic50-limit 1000
