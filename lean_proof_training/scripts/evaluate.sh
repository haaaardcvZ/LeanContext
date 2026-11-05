#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 || ( "$1" != "llada" && "$1" != "llama31" ) ]]; then
  echo "usage: bash scripts/evaluate.sh {llada|llama31} {C0|CMath|CEnv|CFull}" >&2
  exit 2
fi
case "$2" in C0|CMath|CEnv|CFull) ;; *) exit 2;; esac
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
: "${DATASET_ROOT:?Set DATASET_ROOT}"
: "${MATHLIB_ROOT:?Set MATHLIB_ROOT to the built pinned Mathlib checkout}"
: "${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT}"
: "${EVAL_ROOT:?Set EVAL_ROOT}"
: "${MODEL_CACHE:?Set MODEL_CACHE}"
bash evaluation/run.sh "configs/evaluation/${1}_${2}.yaml" "${NUM_GPUS:-16}"
