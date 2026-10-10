#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
MODEL_PATH="${MODEL_PATH:-checkpoints/Qwen3.5-9B}"

python -m src.inference \
  --config configs/inference_config.yaml \
  --input data/test_activity.jsonl \
  --output outputs/predictions_activity.jsonl \
  --model-path "$MODEL_PATH" \
  "$@"

python -m src.metrics \
  --predictions outputs/predictions_activity.jsonl \
  --output outputs/metrics_activity.json

python -m src.analysis \
  --predictions outputs/predictions_activity.jsonl \
  --output-dir outputs/analysis_activity
