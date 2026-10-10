#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

python -m src.metrics \
  --predictions outputs/predictions.jsonl \
  --output outputs/metrics.json
