#!/usr/bin/env bash
# Train ReCord (replay) + curriculum sampler.
#
# Usage:
#   ./scripts/train_record_curriculum_seeds.sh [GPU_ID]
#   ./scripts/train_record_curriculum_seeds.sh 1
#   SEEDS="42" CURRICULUM_STEPS=100 ./scripts/train_record_curriculum_seeds.sh 0
#
# Expects merged curriculum collect:
#   $POP_PATH/saved/other_actions_actions.npy
#   $POP_PATH/saved/difficulty_types.npy
set -euo pipefail

GPU_ID="${1:-0}"
DATA_DIR="${DATA_DIR:-/data/puffer/experiments/record-curriculum-wandb/}"
POP_PATH="${POP_PATH:-/data/puffer/popul_curriculum}"
TYPES_PATH="${TYPES_PATH:-${POP_PATH}/saved/difficulty_types.npy}"
CURRICULUM_STEPS="${CURRICULUM_STEPS:-10000}"
NUM_COMBINATION="${NUM_COMBINATION:-10}"
SEEDS=(${SEEDS:-42 3 11})

if [[ ! -f "${TYPES_PATH}" ]]; then
  echo "Missing ${TYPES_PATH}" >&2
  echo "Collect then merge first:" >&2
  echo "  NUM_CKPTS=4 ./analyze/collect_curriculum_rollout.sh ${GPU_ID}" >&2
  echo "  python analyze/data_concat.py --population-path ${POP_PATH} --total-rollouts 50" >&2
  exit 1
fi
if [[ ! -f "${POP_PATH}/saved/other_actions_actions.npy" ]]; then
  echo "Missing ${POP_PATH}/saved/other_actions_actions.npy (run data_concat.py)" >&2
  exit 1
fi

echo "========== ReCord + curriculum =========="
echo "  gpu=${GPU_ID}"
echo "  data_dir=${DATA_DIR}"
echo "  population=${POP_PATH}"
echo "  types=${TYPES_PATH}"
echo "  curriculum_steps=${CURRICULUM_STEPS}  num_combination=${NUM_COMBINATION}"
echo "  seeds=${SEEDS[*]}"
echo "========================================"

for SEED in "${SEEDS[@]}"; do
  echo ""
  echo ">>> seed=${SEED}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" puffer train_pbt puffer_drive_pbt \
    --pbt.pbt-mode replay \
    --pbt.strategy curriculum \
    --pbt.agent-sampling True \
    --pbt.population-path "${POP_PATH}" \
    --pbt.curriculum-types-path "${TYPES_PATH}" \
    --pbt.curriculum-steps "${CURRICULUM_STEPS}" \
    --pbt.num-combination "${NUM_COMBINATION}" \
    --train.data-dir "${DATA_DIR}" \
    --train.seed "${SEED}" \
    --vec.seed "${SEED}" \
    --wandb \
    --wandb-entity "cilab-ma" \
    --wandb-group "record-curriculum-seed-${SEED}" \
    --wandb-project "puffer-drive-icra"
done

echo ""
echo "All seeds finished. Checkpoints under: ${DATA_DIR}"
