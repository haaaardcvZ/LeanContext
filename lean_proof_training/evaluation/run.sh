#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 CONFIG.yaml [NUM_GPUS]" >&2
  exit 2
fi

CONFIG="$1"
NUM_GPUS="${2:-16}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

python -m evaluation.source_index --config "$CONFIG"
python -m evaluation.gold_preflight --config "$CONFIG"
torchrun --standalone --nproc_per_node "$NUM_GPUS" --module evaluation.generate --config "$CONFIG"
python -m evaluation.verify --config "$CONFIG"
