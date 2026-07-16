#!/usr/bin/env bash
# Train one SAE per experiment on collected activations.npz
#
# Usage:
#   ./analyze/sae/run_train_sae.sh
#   ARCH=standard EXPERIMENTS=selfplay GPU_ID=0 ./analyze/sae/run_train_sae.sh
#   NUM_STEPS=5000 OUT_DIR=/data/puffer/sae/runs/smoke ./analyze/sae/run_train_sae.sh
#
# Writes:
#   <OUT_DIR>/selfplay/sae_{best,last}.pt
#   <OUT_DIR>/reactive_0.25/...
#   <OUT_DIR>/replay_0.25/...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"

SAE_ROOT="${SAE_ROOT:-/data/puffer/sae}"
PROBE_STEP="${PROBE_STEP:-1908}"
EXPERIMENTS="${EXPERIMENTS:-selfplay,reactive_0.25,replay_0.25}"
ARCH="${ARCH:-topk}"
EXPANSION="${EXPANSION:-2}"
K="${K:-8}"
L1="${L1:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
NUM_STEPS="${NUM_STEPS:-1000}"
LR="${LR:-5e-4}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-1000}"
OUT_DIR="${OUT_DIR:-/data/puffer/sae/runs/${ARCH}_exp${EXPANSION}_k${K}_step${PROBE_STEP}}"

mkdir -p "$OUT_DIR"

echo "Train one SAE per experiment"
echo "  data: ${SAE_ROOT}/human_replay/{training,validation}/step_$(printf '%06d' "$PROBE_STEP")/activations.npz"
echo "  experiments=${EXPERIMENTS}"
echo "  arch=${ARCH} expansion=${EXPANSION} k=${K} steps=${NUM_STEPS}"
echo "  checkpoints: every ${SAVE_EVERY} + ${CHECKPOINT_STEPS}"
echo "  out=${OUT_DIR}/<experiment>/"

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$SCRIPT_DIR/train_sae.py" \
  --sae-root "$SAE_ROOT" \
  --probe-step "$PROBE_STEP" \
  --experiments "$EXPERIMENTS" \
  --architecture "$ARCH" \
  --expansion "$EXPANSION" \
  --k "$K" \
  --l1-coefficient "$L1" \
  --batch-size "$BATCH_SIZE" \
  --num-steps "$NUM_STEPS" \
  --lr "$LR" \
  --device "$DEVICE" \
  --save-every "$SAVE_EVERY" \
  --checkpoint-steps "$CHECKPOINT_STEPS" \
  --out-dir "$OUT_DIR"
