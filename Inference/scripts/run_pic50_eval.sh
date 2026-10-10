#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
MODEL_PATH="${MODEL_PATH:-checkpoints/Qwen3.5-9B}"

python -m src.inference \
  --config configs/inference_config.yaml \
  --input data/test_pic50.jsonl \
  --output outputs/predictions_pic50.jsonl \
  --model-path "$MODEL_PATH" \
  "$@"

python -m src.metrics \
  --predictions outputs/predictions_pic50.jsonl \
  --output outputs/metrics_pic50.json

python -m src.analysis \
  --predictions outputs/predictions_pic50.jsonl \
  --output-dir outputs/analysis_pic50
