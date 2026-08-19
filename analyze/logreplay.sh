#!/usr/bin/env bash
# Human log-replay eval. Ego checkpoints default to scripts/train_pbt_seeds.sh DATA_DIR.
#
# TRAIN_MODE  record | reactive | all
#             same as scripts/train_pbt_seeds.sh MODE (MODE= is accepted as an alias)
#
# Usage:
#   STRATEGY=uniform POP_PATH=lane_nominal ./analyze/logreplay.sh 0
#   TRAIN_MODE=record STRATEGY=uniform ./analyze/logreplay.sh 0
#   MODE=reactive STRATEGY=uniform ./analyze/logreplay.sh 0
#   ./analyze/logreplay.sh 0 replay_uniform_nominal
#
# Defaults:
#   TRAIN_MODE=all → record + reactive ego
#
# POP_PATH: suffix after popul_ (lane_nominal | mix | curriculum | ...)
# FOLDER (arg 2) pins a single ego dir and skips the TRAIN_MODE loop.
set -euo pipefail
GPU_ID=${1:-0}
NUM_MAPS="${NUM_MAPS:-10000}"

# train_pbt_seeds.sh MODE=record|reactive
TRAIN_MODE="${TRAIN_MODE:-${MODE:-all}}"
STRATEGY="${STRATEGY:-uniform}"
case "${STRATEGY}" in
  plr) STRATEGY="prioritized" ;;
esac

if [[ "${STRATEGY}" == "curriculum" ]]; then
  POP_PATH="${POP_PATH:-${POPULATION_PATH:-curriculum}}"
else
  POP_PATH="${POP_PATH:-${POPULATION_PATH:-lane_nominal}}"
fi
pop_key="$(basename "${POP_PATH%/}")"
pop_key="${pop_key#popul_}"
POP_PATH="/data/puffer/popul_${pop_key}"
POP_NAME="popul_${pop_key}"
POP_SHORT="${pop_key##*_}"

pick_ego_folder() {
  local preferred="$1" legacy="$2"
  local root="/data/puffer/experiments"
  if compgen -G "${root}/${preferred}/puffer_drive_*.pt" > /dev/null; then
    echo "${preferred}"
    return
  fi
  if compgen -G "${root}/${legacy}/puffer_drive_*.pt" > /dev/null; then
    echo "${legacy}"
    return
  fi
  echo "${preferred}"
}

resolve_train_mode() {
  local tm="$1"
  case "${tm}" in
    record)
      PBT_MODE="replay"
      TRAIN_MODE_LEGACY="record"
      ;;
    reactive)
      PBT_MODE="reactive"
      TRAIN_MODE_LEGACY="reactive"
      ;;
    *)
      echo "TRAIN_MODE must be record, reactive, or all (got: ${tm})" >&2
      echo "  (this is scripts/train_pbt_seeds.sh MODE)" >&2
      exit 1
      ;;
  esac
}

FOLDER_ARG="${2:-}"

case "${TRAIN_MODE}" in
  all|both) TRAIN_MODE_LIST=(record reactive) ;;
  record) TRAIN_MODE_LIST=(record) ;;
  reactive) TRAIN_MODE_LIST=(reactive) ;;
  replay)
    echo "Use TRAIN_MODE=record (train MODE). replay is a zero-shot EVAL_MODE, not log-replay." >&2
    exit 1
    ;;
  *)
    echo "TRAIN_MODE must be record, reactive, or all (got: ${TRAIN_MODE})" >&2
    echo "  TRAIN_MODE = scripts/train_pbt_seeds.sh MODE" >&2
    exit 1
    ;;
esac
if [[ -n "${FOLDER_ARG}" ]]; then
  TRAIN_MODE_LIST=(pinned)
fi

run_one() {
  local -a EGOS=()
  local f bn id EGO
  shopt -s nullglob
  for f in /data/puffer/experiments/${FOLDER}/puffer_drive_*.pt; do
    bn=$(basename "$f")
    id=${bn#puffer_drive_}
    id=${id%.pt}
    EGOS+=("$id")
  done
  shopt -u nullglob

  echo "========== log-replay =========="
  echo "  train_mode=${tm}  folder=${FOLDER}"
  echo "  num_maps=${NUM_MAPS}"
  echo "  egos=${EGOS[*]-}"
  echo "================================"

  if [[ ${#EGOS[@]} -eq 0 ]]; then
    echo "Skip: no ego checkpoints under /data/puffer/experiments/${FOLDER}/" >&2
    return 0
  fi

  for EGO in "${EGOS[@]}"; do
    echo "Running log-replay: EGO ${EGO}"
    CUDA_VISIBLE_DEVICES=$GPU_ID puffer eval puffer_drive \
      --eval.human-replay-eval True \
      --eval.human-replay-save-results True \
      --env.termination-mode "0" \
      --eval.wosac-num-maps "${NUM_MAPS}" \
      --load-model-path "/data/puffer/experiments/${FOLDER}/puffer_drive_${EGO}.pt"
  done
}

echo "TRAIN_MODE=${TRAIN_MODE_LIST[*]}  POP=${POP_NAME}  STRATEGY=${STRATEGY}"

for tm in "${TRAIN_MODE_LIST[@]}"; do
  if [[ -n "${FOLDER_ARG}" ]]; then
    FOLDER="${FOLDER_ARG}"
  else
    resolve_train_mode "${tm}"
    preferred_folder="${PBT_MODE}_${STRATEGY}_${POP_SHORT}"
    legacy_folder="${TRAIN_MODE_LEGACY}-${STRATEGY}-${POP_NAME}-wandb"
    FOLDER="$(pick_ego_folder "${preferred_folder}" "${legacy_folder}")"
  fi
  run_one
done
