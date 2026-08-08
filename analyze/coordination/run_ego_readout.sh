#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
cd "$REPO_ROOT"

GPU_ID="${1:-${GPU_ID:-0}}"
NUM_MAPS="${NUM_MAPS:-600}"
MAX_SEEDS="${MAX_SEEDS:-3}"
DEVICE="${DEVICE:-cuda}"
OUT_ROOT="${OUT_ROOT:-/data/puffer/results/coordination}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-/data/puffer/experiments}"
REUSE_PACKS="${REUSE_PACKS:-0}"

reuse_flag=()
if [[ "$REUSE_PACKS" == "1" ]]; then
  reuse_flag+=(--reuse-packs)
fi

echo "========== Ego readout (divergence-conditioned) =========="
echo "  GPU=${GPU_ID}  maps=${NUM_MAPS}  seeds=${MAX_SEEDS}"
echo "  out=${OUT_ROOT}/ego_readout/"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$SCRIPT_DIR/ego_readout.py" \
  --experiments-root "$EXPERIMENTS_ROOT" \
  --out-root "$OUT_ROOT" \
  --num-maps "$NUM_MAPS" \
  --max-seeds "$MAX_SEEDS" \
  --device "$DEVICE" \
  "${reuse_flag[@]}"

echo
echo "Primary: ${OUT_ROOT}/ego_readout/summary.json"
