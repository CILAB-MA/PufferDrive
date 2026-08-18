#!/usr/bin/env bash
# Train a diverse self-play population with MEP Population Entropy bonus.
#
# Usage:
#   ./analyze/generate_population.sh [GPU_ID]
#   OUT=/data/puffer/popul_mep K=4 ENT_POOL=0.01 ./analyze/generate_population.sh 0
set -euo pipefail

GPU_ID="${1:-0}"
OUT="${OUT:-/data/puffer/popul_mep}"
K="${K:-4}"
ENT_POOL="${ENT_POOL:-0.01}"
SEED="${SEED:-0}"

echo "========== generate_population (MEP) =========="
echo "  gpu=${GPU_ID}"
echo "  out=${OUT}"
echo "  k=${K}  ent_pool=${ENT_POOL}  seed=${SEED}"
echo "=============================================="

CUDA_VISIBLE_DEVICES="${GPU_ID}" puffer generate_population puffer_drive \
  --pbt.generate-out-path "${OUT}" \
  --pbt.generate-k "${K}" \
  --pbt.generate-ent-pool "${ENT_POOL}" \
  --pbt.generate-seed "${SEED}"
