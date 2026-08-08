#!/usr/bin/env bash
# Canonical coordination analysis: divergence-conditioned Rec vs Rea vs SP.
#
# Usage:
#   ./analyze/coordination/run_coordination.sh
#   ./analyze/coordination/run_coordination.sh 1
#   NUM_MAPS=300 MAX_SEEDS=1 ./analyze/coordination/run_coordination.sh
#   REUSE_PACKS=1 ./analyze/coordination/run_coordination.sh

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

echo "========== Coordination (divergence-first) =========="
echo "  GPU=${GPU_ID}  maps=${NUM_MAPS}  seeds=${MAX_SEEDS}"
echo "  out=${OUT_ROOT}/divergence_scenes/"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$SCRIPT_DIR/divergence.py" \
  --experiments-root "$EXPERIMENTS_ROOT" \
  --out-root "$OUT_ROOT" \
  --num-maps "$NUM_MAPS" \
  --max-seeds "$MAX_SEEDS" \
  --device "$DEVICE" \
  "${reuse_flag[@]}"

echo
echo "Primary outputs:"
echo "  ${OUT_ROOT}/divergence_scenes/summary.json"
echo "  ${OUT_ROOT}/divergence_scenes/packs/seed*_{record,reactive,selfplay}.npz"
