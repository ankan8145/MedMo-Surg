#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

python -m src.inference \
  --config configs/inference_config.yaml \
  --input data/sample_inputs.jsonl \
  --output outputs/predictions.jsonl \
  "$@"
