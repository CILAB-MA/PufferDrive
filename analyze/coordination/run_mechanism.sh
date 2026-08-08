#!/usr/bin/env bash
# Brake-logit bias mechanism on Reactive.
#   MODE=single (default) → mechanism_patch/
#   MODE=sweep            → mechanism_deep/ (dose/window/approach + R+)
#
#   ./analyze/coordination/run_mechanism.sh 0
#   MODE=sweep ./analyze/coordination/run_mechanism.sh 0

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
cd "$REPO_ROOT"

GPU_ID="${1:-${GPU_ID:-0}}"
MODE="${MODE:-single}"
NUM_MAPS="${NUM_MAPS:-2000}"
MAX_SEEDS="${MAX_SEEDS:-3}"
DEVICE="${DEVICE:-cuda}"
OUT_ROOT="${OUT_ROOT:-/data/puffer/results/coordination}"
EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-/data/puffer/experiments}"
REUSE_BASELINE="${REUSE_BASELINE:-0}"
BIAS="${BIAS:-1.5}"
BASELINE_PACK_DIR="${BASELINE_PACK_DIR:-}"
SHARD_SIZE="${SHARD_SIZE:-1000}"

reuse_flag=()
if [[ "$REUSE_BASELINE" == "1" ]]; then
  reuse_flag+=(--reuse-baseline)
fi
baseline_flag=()
if [[ -n "$BASELINE_PACK_DIR" ]]; then
  baseline_flag+=(--baseline-pack-dir "$BASELINE_PACK_DIR")
fi

if [[ "$MODE" == "sweep" ]]; then
  OUT_SUB="mechanism_deep"
else
  OUT_SUB="mechanism_patch"
fi

echo "========== Mechanism (${MODE}) =========="
echo "  GPU=${GPU_ID}  maps=${NUM_MAPS}  seeds=${MAX_SEEDS}  bias=${BIAS}  shard=${SHARD_SIZE}"
echo "  out=${OUT_ROOT}/${OUT_SUB}/"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -u "$SCRIPT_DIR/mechanism.py" \
  --mode "$MODE" \
  --experiments-root "$EXPERIMENTS_ROOT" \
  --out-root "$OUT_ROOT" \
  --num-maps "$NUM_MAPS" \
  --max-seeds "$MAX_SEEDS" \
  --device "$DEVICE" \
  --brake-logit-bias "$BIAS" \
  --shard-size "$SHARD_SIZE" \
  "${reuse_flag[@]}" \
  "${baseline_flag[@]}"

echo
echo "Primary: ${OUT_ROOT}/${OUT_SUB}/summary.json"
