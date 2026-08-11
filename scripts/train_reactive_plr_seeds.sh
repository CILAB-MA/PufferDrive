#!/usr/bin/env bash
# Train Reactive + PLR (prioritized) for 3 seeds.
#
# Usage:
#   ./scripts/train_reactive_plr_seeds.sh [GPU_ID]
#   ./scripts/train_reactive_plr_seeds.sh 1
#   DATA_DIR=/data/puffer/experiments/reactive-plr-wandb/ \
#     POPULATION_PATH=/data/puffer/popul_lane_nominal/ \
#     ./scripts/train_reactive_plr_seeds.sh 0
#   SEEDS="42 55 1" ./scripts/train_reactive_plr_seeds.sh
set -euo pipefail

GPU_ID="${1:-0}"
DATA_DIR="${DATA_DIR:-/data/puffer/experiments/reactive-plr-wandb/}"
POPULATION_PATH="${POPULATION_PATH:-/data/puffer/popul_lane_nominal/}"
read -r -a SEED_LIST <<< "${SEEDS:-42 3 11}"

if [[ ! -d "${POPULATION_PATH}" ]]; then
  echo "Population directory not found: ${POPULATION_PATH}" >&2
  exit 1
fi
if ! compgen -G "${POPULATION_PATH%/}/*.pt" > /dev/null; then
  echo "No reactive policy checkpoints (*.pt) found in: ${POPULATION_PATH}" >&2
  exit 1
fi
if [[ ! -f "${POPULATION_PATH%/}/saved/global_ids.npy" ]]; then
  echo "Reactive PLR requires: ${POPULATION_PATH%/}/saved/global_ids.npy" >&2
  exit 1
fi

echo "========== Reactive + PLR (3 seeds) =========="
echo "  gpu=${GPU_ID}"
echo "  data_dir=${DATA_DIR}"
echo "  population_path=${POPULATION_PATH}"
echo "  seeds=${SEED_LIST[*]}"
echo "============================================"

for SEED in "${SEED_LIST[@]}"; do
  echo ""
  echo ">>> seed=${SEED}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" puffer train_pbt puffer_drive_pbt \
    --pbt.pbt-mode reactive \
    --pbt.population-path "${POPULATION_PATH}" \
    --pbt.strategy prioritized \
    --pbt.agent-sampling True \
    --pbt.score-transform rank_low \
    --train.data-dir "${DATA_DIR}" \
    --train.seed "${SEED}" \
    --vec.seed "${SEED}" \
    --eval.human-replay-eval True \
    --wandb \
    --wandb-entity "cilab-ma" \
    --wandb-group "reactive-plr-seed-${SEED}" \
    --wandb-project "puffer-drive-icra"
done

echo ""
echo "All seeds finished. Checkpoints under: ${DATA_DIR}"
