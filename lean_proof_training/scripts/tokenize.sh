#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "llada" && "$1" != "llama31" && "$1" != "both" ) ]]; then
  echo "usage: bash scripts/tokenize.sh [llada|llama31|both]" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR/.."
: "${DATASET_RELEASE_ROOT:?Set DATASET_RELEASE_ROOT to the dataset release directory}"
: "${DATASET_ROOT:?Set DATASET_ROOT to a new runtime directory used by the training YAMLs}"
MODELS=(llada llama31)
if [[ "${1:-both}" != "both" ]]; then MODELS=("$1"); fi
# Saved paper-format views are exported without reselection. Preparation writes
# the exact paths consumed by the training YAMLs and independently audits them.
"${PYTHON_BIN:-python3}" -m dataset_preparation.prepare_runtime \
  --dataset-root "$DATASET_RELEASE_ROOT" --output-root "$DATASET_ROOT" \
  --models "${MODELS[@]}" --execute
