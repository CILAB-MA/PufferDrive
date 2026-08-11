#!/usr/bin/env bash
# Train ReCord (replay) + PLR (prioritized) for 3 seeds.
#
# Usage:
#   ./scripts/train_record_plr_seeds.sh [GPU_ID]
#   ./scripts/train_record_plr_seeds.sh 1
#   DATA_DIR=/data/puffer/experiments/icra/ ./scripts/train_record_plr_seeds.sh 0
#   SEEDS="42 55 1" ./scripts/train_record_plr_seeds.sh
set -euo pipefail

GPU_ID="${1:-0}"
DATA_DIR="${DATA_DIR:-/data/puffer/experiments/record-plr-wandb/}"
SEEDS=(${SEEDS:-42 3 11})

echo "========== ReCord + PLR (3 seeds) =========="
echo "  gpu=${GPU_ID}"
echo "  data_dir=${DATA_DIR}"
echo "  seeds=${SEEDS[*]}"
echo "============================================"

for SEED in "${SEEDS[@]}"; do
  echo ""
  echo ">>> seed=${SEED}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" puffer train_pbt puffer_drive_pbt \
    --pbt.pbt-mode replay \
    --pbt.strategy prioritized \
    --pbt.agent-sampling True \
    --pbt.score-transform rank_low \
    --train.data-dir "${DATA_DIR}" \
    --train.seed "${SEED}" \
    --vec.seed "${SEED}" \
    --eval.human-replay-eval True \
    --wandb \
    --wandb-entity "cilab-ma" \
    --wandb-group "record-plr-seed-${SEED}" \
    --wandb-project "puffer-drive-icra"
done

echo ""
echo "All seeds finished. Checkpoints under: ${DATA_DIR}"
