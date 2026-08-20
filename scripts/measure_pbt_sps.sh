#!/usr/bin/env bash
# Drive-PBT SPS benchmark: 30M agent steps, one environment worker.
#
# Usage:
#   MODE=record STRATEGY=uniform ./scripts/measure_pbt_sps.sh [GPU_ID] [SEED]
#
# MODE: record | reactive
# STRATEGY: uniform | prioritized | curriculum
set -euo pipefail

GPU_ID="${1:-0}"
SEED="${2:-42}"
MODE="${MODE:-record}"
STRATEGY="${STRATEGY:-uniform}"
NUM_COMBINATION="${NUM_COMBINATION:-10}"
SCORE_TRANSFORM="${SCORE_TRANSFORM:-rank_low}"
CURRICULUM_STEPS="${CURRICULUM_STEPS:-16}"

case "${MODE}" in
  record) PBT_MODE="replay" ;;
  reactive) PBT_MODE="reactive" ;;
  *)
    echo "MODE must be record or reactive (got: ${MODE})" >&2
    exit 1
    ;;
esac

case "${STRATEGY}" in
  uniform|prioritized|curriculum) ;;
  *)
    echo "STRATEGY must be uniform, prioritized, or curriculum (got: ${STRATEGY})" >&2
    exit 1
    ;;
esac

if [[ "${STRATEGY}" == "curriculum" ]]; then
  POP_PATH="${POP_PATH:-/data/puffer/popul_curriculum}"
else
  POP_PATH="${POP_PATH:-/data/puffer/popul_lane_nominal}"
fi
POP_PATH="${POP_PATH%/}"
TYPES_PATH="${TYPES_PATH:-${POP_PATH}/saved/difficulty_types.npy}"
POP_NAME="$(basename "${POP_PATH}")"
EXP_NAME="sps_${PBT_MODE}_${STRATEGY}_${POP_NAME}_w1_e4_b4_seed${SEED}"
DATA_DIR="/data/puffer/experiments/${EXP_NAME}"
LOG_DIR="/data/puffer/sps_logs"

if [[ ! -f "${POP_PATH}/saved/global_ids.npy" ]]; then
  echo "Missing ${POP_PATH}/saved/global_ids.npy" >&2
  exit 1
fi
if [[ "${PBT_MODE}" == "replay" && ! -f "${POP_PATH}/saved/other_actions_actions.npy" ]]; then
  echo "Missing ${POP_PATH}/saved/other_actions_actions.npy" >&2
  exit 1
fi
if [[ "${PBT_MODE}" == "reactive" && ! -f "${POP_PATH}/saved/population_keys.npy" ]]; then
  echo "Missing ${POP_PATH}/saved/population_keys.npy" >&2
  exit 1
fi
if [[ "${STRATEGY}" == "curriculum" && ! -f "${TYPES_PATH}" ]]; then
  echo "Missing ${TYPES_PATH}" >&2
  exit 1
fi

mkdir -p "${DATA_DIR}" "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S).log"

CMD=(
  puffer train_pbt puffer_drive_pbt
  --pbt.pbt-mode "${PBT_MODE}"
  --pbt.population-path "${POP_PATH}"
  --pbt.strategy "${STRATEGY}"
  --pbt.num-combination "${NUM_COMBINATION}"
  --train.total-timesteps 30000000
  --train.data-dir "${DATA_DIR}"
  --train.seed "${SEED}"
  --vec.backend Multiprocessing
  --vec.num-workers 1
  --vec.num-envs 4
  --vec.batch-size 4
  --vec.seed "${SEED}"
  --eval.human-replay-eval False
)

if [[ "${STRATEGY}" == "prioritized" ]]; then
  CMD+=(--pbt.score-transform "${SCORE_TRANSFORM}")
elif [[ "${STRATEGY}" == "curriculum" ]]; then
  CMD+=(--pbt.curriculum-types-path "${TYPES_PATH}")
  CMD+=(--pbt.curriculum-steps "${CURRICULUM_STEPS}")
fi

echo "Drive-PBT SPS: mode=${PBT_MODE} strategy=${STRATEGY} seed=${SEED}"
echo "Vector: Multiprocessing workers=1 envs=4 batch=4, steps=30000000"
echo "Log: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
