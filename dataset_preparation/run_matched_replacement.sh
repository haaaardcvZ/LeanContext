#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: bash dataset_preparation/run_matched_replacement.sh DATASET_ROOT NEW_OUTPUT_ROOT" >&2
  exit 2
fi
DATASET_ROOT="$1"
OUTPUT_ROOT="$2"
CALLER_ROOT="$(pwd -P)"
[[ "$DATASET_ROOT" = /* ]] || DATASET_ROOT="$CALLER_ROOT/$DATASET_ROOT"
[[ "$OUTPUT_ROOT" = /* ]] || OUTPUT_ROOT="$CALLER_ROOT/$OUTPUT_ROOT"
CODE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$CODE_ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -m dataset_preparation.matched_replacement \
  --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT" \
  --sample-size 2500 --final-size 2000 --seed 2026 --input-limit 1024
"$PYTHON_BIN" -m dataset_preparation.validate_replacement \
  --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT"
"$PYTHON_BIN" -m dataset_preparation.finalize_replacement \
  --dataset-root "$DATASET_ROOT" --output-root "$OUTPUT_ROOT"
