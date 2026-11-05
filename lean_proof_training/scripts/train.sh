#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ( "$1" != "dllm" && "$1" != "llama31" ) ]]; then
  echo "usage: $0 {dllm|llama31} CONFIG.yaml" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
NUM_GPUS="${NUM_GPUS:-16}"
if (( NUM_GPUS <= 0 || 32 % NUM_GPUS != 0 )); then
  echo "NUM_GPUS must be a positive divisor of effective batch size 32" >&2
  exit 2
fi
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-$((32 / NUM_GPUS))}"
torchrun --standalone --nproc_per_node="$NUM_GPUS" \
  -m "lean_proof_training.$1.train" --config "$2" \
  --set "distributed.expected_processes=$NUM_GPUS" \
  --set "training.gradient_accumulation_steps=$GRAD_ACCUM_STEPS"
