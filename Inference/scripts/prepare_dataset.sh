#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

python -m src.dataset_prepare "$@"
