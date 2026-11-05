#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != "llada" && "$1" != "llama31" ) ]]; then
  echo "usage: bash scripts/run_baselines.sh {llada|llama31}" >&2
  exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
ARCH="$1"
if [[ "$ARCH" == "llada" ]]; then ARCH="dllm"; fi
for view in C0 CMath CEnv CFull; do
  bash scripts/train.sh "$ARCH" "configs/$1/full_sft_${view}.yaml"
done
