#!/usr/bin/env bash
# HEICO-FOCUS SFT launcher.  Usage: scripts/train_heico_focus_sft.sh {smoke|train}
# All paths are relative to the trl/ directory and can be overridden via env vars.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-train}"
MODEL_PATH="${MODEL_PATH:-checkpoints/MedMO-9B-SFT}"
DATASET_DIR="${DATASET_DIR:-data/heico_focus}"   # holds train/ and test/ splits
OUTPUT_DIR="${OUTPUT_DIR:-outputs/MedMO-9B-HeiCo-SFT}"
CONFIG="${CONFIG:-configs/heico_focus_sft.yaml}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-trl/accelerate_configs/zero3.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

EXTRA=()
[[ "$MODE" == "smoke" ]] && EXTRA+=(--max_steps 1 --eval_strategy no --save_strategy no)

accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes "$NUM_PROCESSES" \
  --main_process_port "$MAIN_PROCESS_PORT" \
  trl/scripts/sft.py \
  --config "$CONFIG" \
  --model_name_or_path "$MODEL_PATH" \
  --dataset_name "$DATASET_DIR" \
  --output_dir "$OUTPUT_DIR" \
  "${EXTRA[@]}"
