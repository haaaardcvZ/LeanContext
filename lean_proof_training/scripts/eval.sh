#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ( "$1" != "dllm" && "$1" != "llama31" ) ]]; then
  echo "usage: $0 {dllm|llama31} CONFIG.yaml" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
NUM_GPUS="${NUM_GPUS:-16}"
torchrun --standalone --nproc_per_node="$NUM_GPUS" \
  -m "lean_proof_training.$1.eval" --config "$2" \
  --set "distributed.expected_processes=$NUM_GPUS"
