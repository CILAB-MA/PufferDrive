#!/usr/bin/env bash
# Zero-shot eval. Ego checkpoints default to scripts/train_pbt_seeds.sh DATA_DIR.
#
# Knobs (do not mix these up):
#   TRAIN_MODE  record | reactive | all
#               same as scripts/train_pbt_seeds.sh MODE (MODE= is accepted as an alias)
#   EVAL_MODE   replay | reactive-play | both
#               zero-shot protocol: frozen-other replay vs live reactive-play
#   UNSEEN_MODE unseen_other_rewards | unseen_other_seeds | all
#   DATA_MODE   training | validation  (default: validation — eval map corpus)
#   MAP_DIR     absolute path; overrides DATA_MODE
#
# Usage:
#   STRATEGY=uniform POP_PATH=lane_nominal ./analyze/zero_shot.sh 0
#   TRAIN_MODE=record STRATEGY=uniform EVAL_MODE=replay ./analyze/zero_shot.sh 0
#   MODE=reactive STRATEGY=uniform EVAL_MODE=both ./analyze/zero_shot.sh 0
#   ./analyze/zero_shot.sh 0 replay_uniform_nominal popul_lane_nominal unseen_other_seeds both
#   DATA_MODE=validation ./analyze/zero_shot.sh 0 human_uniform_nominal popul_nominal all both
#   DATA_MODE=training ./analyze/zero_shot.sh 0 human_uniform_nominal popul_nominal all both
#
# Defaults:
#   TRAIN_MODE=all   → record + reactive ego
#   UNSEEN_MODE=all  → unseen_other_rewards + unseen_other_seeds
#   EVAL_MODE=both   → replay + reactive-play
#   DATA_MODE=validation → /data/puffer/resources/drive/binaries/validation
#
# POP_PATH: suffix after popul_ (lane_nominal | mix | curriculum | ...)
#   → /data/puffer/popul_${POP_PATH}
# FOLDER (arg 2) pins a single ego dir and skips the TRAIN_MODE loop.
# RESULTS_FOLDER (env, optional): write under /data/puffer/results/${RESULTS_FOLDER}
#   while still loading egos from experiments/${FOLDER}. Defaults to FOLDER.
set -euo pipefail
GPU_ID=${1:-0}
DRIVE_BINARIES_ROOT="${DRIVE_BINARIES_ROOT:-/data/puffer/resources/drive/binaries}"

resolve_map_dir() {
  if [[ -n "${MAP_DIR:-}" ]]; then
    echo "${MAP_DIR}"
    return
  fi
  local mode="${DATA_MODE:-validation}"
  case "${mode}" in
    training|validation)
      echo "${DRIVE_BINARIES_ROOT}/${mode}"
      ;;
    *)
      echo "DATA_MODE must be training or validation (got: ${mode})" >&2
      exit 1
      ;;
  esac
}

EVAL_MAP_DIR="$(resolve_map_dir)"
if [[ ! -d "${EVAL_MAP_DIR}" ]]; then
  echo "Missing map_dir: ${EVAL_MAP_DIR}" >&2
  exit 1
fi

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
# Always /data/puffer/popul_<suffix> (lane_nominal | mix | curriculum | ...)
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
      echo "  (this is scripts/train_pbt_seeds.sh MODE; use EVAL_MODE for replay vs reactive-play)" >&2
      exit 1
      ;;
  esac
}

FOLDER_ARG="${2:-}"
POPULATION_MODE=${3:-${POP_NAME}}
UNSEEN_MODE="${UNSEEN_MODE:-${4:-all}}"
EVAL_MODE="${EVAL_MODE:-${5:-both}}"
RESULTS_ROOT="${6:-/data/puffer/results_new}"

case "${TRAIN_MODE}" in
  all|both) TRAIN_MODE_LIST=(record reactive) ;;
  record) TRAIN_MODE_LIST=(record) ;;
  reactive) TRAIN_MODE_LIST=(reactive) ;;
  replay)
    echo "TRAIN_MODE=replay is the eval protocol. Use TRAIN_MODE=record (train MODE) and EVAL_MODE=replay." >&2
    exit 1
    ;;
  *)
    echo "TRAIN_MODE must be record, reactive, or all (got: ${TRAIN_MODE})" >&2
    echo "  TRAIN_MODE = scripts/train_pbt_seeds.sh MODE" >&2
    echo "  EVAL_MODE  = replay | reactive-play | both" >&2
    exit 1
    ;;
esac
if [[ -n "${FOLDER_ARG}" ]]; then
  TRAIN_MODE_LIST=(pinned)
fi

case "${UNSEEN_MODE}" in
  all|both) UNSEEN_LIST=(unseen_other_rewards unseen_other_seeds) ;;
  unseen_other_rewards|unseen_other_seeds) UNSEEN_LIST=("${UNSEEN_MODE}") ;;
  *)
    echo "UNSEEN_MODE must be unseen_other_rewards, unseen_other_seeds, or all (got: ${UNSEEN_MODE})" >&2
    exit 1
    ;;
esac

case "${EVAL_MODE}" in
  replay) EVAL_LIST=(replay) ;;
  reactive|reactive-play) EVAL_LIST=(reactive-play) ;;
  both|all) EVAL_LIST=(replay reactive-play) ;;
  *)
    echo "EVAL_MODE must be replay, reactive-play, or both (got: ${EVAL_MODE})" >&2
    echo "  EVAL_MODE is the zero-shot protocol, not train MODE/TRAIN_MODE." >&2
    exit 1
    ;;
esac

run_zeroshot() {
  local ego_id=$1 other_id=$2 zsm=$3
  shift 3
  local -a extra=("$@")
  CUDA_VISIBLE_DEVICES=$GPU_ID puffer zeroshot puffer_drive \
    --load-multiple-model-path \
      "/data/puffer/experiments/${FOLDER}/puffer_drive_${ego_id}.pt" \
      "/data/puffer/${POPULATION_MODE}/${UNSEEN_MODE}/puffer_drive_${other_id}.pt" \
    --zero-shot-mode "${zsm}" \
    --eval.map-dir "${EVAL_MAP_DIR}" \
    --env.termination-mode "0" \
    --env.goal-behavior "3" \
    --pbt.pbt-mode "replay" \
    "${extra[@]}"
}

# Must match evaluator.save_replay / play_replay: path[-11:-3] of other ckpt.
other_buffer_path() {
  local other_id=$1
  local other_ckpt="/data/puffer/${POPULATION_MODE}/${UNSEEN_MODE}/puffer_drive_${other_id}.pt"
  local buf_id=${other_ckpt: -11:8}
  echo "/data/puffer/experiments/${UNSEEN_MODE}/other_action_buffer/other_actions_${buf_id}.npy"
}

# Validation 10k maps → ~60473 agents / ~10000 egos → 50473 other actions.
# Override with EXPECTED_OTHER_AGENTS if map corpus changes.
EXPECTED_OTHER_AGENTS="${EXPECTED_OTHER_AGENTS:-50473}"

# True if buffer exists and first dim matches current eval env other-agent count.
other_buffer_ok() {
  local buf_path=$1
  if [[ ! -f "${buf_path}" ]]; then
    return 1
  fi
  local n
  n="$(
    python3 - "${buf_path}" <<'PY'
import sys
import numpy as np
a = np.load(sys.argv[1], mmap_mode="r")
print(int(a.shape[0]))
PY
  )"
  if [[ "${n}" != "${EXPECTED_OTHER_AGENTS}" ]]; then
    echo "stale buffer shape=${n} (want ${EXPECTED_OTHER_AGENTS}): ${buf_path}" >&2
    return 1
  fi
  return 0
}

run_one() {
  local eval_one="$1"
  local results_folder="${RESULTS_FOLDER:-${FOLDER}}"
  local SCENARIO_LOG_DIR="${RESULTS_ROOT}/${results_folder}/${UNSEEN_MODE}/scenario_logs"
  mkdir -p "${SCENARIO_LOG_DIR}"

  local -a EGOS=() OTHERS=()
  local f bn id
  shopt -s nullglob
  for f in /data/puffer/experiments/${FOLDER}/puffer_drive_*.pt; do
    bn=$(basename "$f")
    id=${bn#puffer_drive_}
    id=${id%.pt}
    EGOS+=("$id")
  done
  for f in /data/puffer/${POPULATION_MODE}/${UNSEEN_MODE}/puffer_drive_*.pt; do
    bn=$(basename "$f")
    id=${bn#puffer_drive_}
    id=${id%.pt}
    OTHERS+=("$id")
  done
  shopt -u nullglob

  echo "========== zeroshot =========="
  echo "  train_mode=${tm}  eval_mode=${eval_one}  unseen=${UNSEEN_MODE}"
  echo "  map_dir=${EVAL_MAP_DIR}  data_mode=${DATA_MODE:-validation}"
  echo "  folder=${FOLDER}  results=${results_folder}"
  echo "  others=/data/puffer/${POPULATION_MODE}/${UNSEEN_MODE}"
  echo "  egos=${EGOS[*]-}"
  echo "  others=${OTHERS[*]-}"
  echo "=============================="

  if [[ ${#EGOS[@]} -eq 0 || ${#OTHERS[@]} -eq 0 ]]; then
    echo "Skip: need ego under experiments/${FOLDER} and others under ${POPULATION_MODE}/${UNSEEN_MODE}." >&2
    return 0
  fi

  local MP1 MP2 ZSM BUF_PATH LOG_PATH REF_EGO

  if [[ "${eval_one}" == "replay" ]]; then
    REF_EGO="${EGOS[0]}"
    for MP2 in "${OTHERS[@]}"; do
      BUF_PATH=$(other_buffer_path "${MP2}")
      if other_buffer_ok "${BUF_PATH}"; then
        echo "skip save-replay: buffer ok other=${MP2} (${BUF_PATH})"
        continue
      fi
      if [[ -f "${BUF_PATH}" ]]; then
        mkdir -p "$(dirname "${BUF_PATH}")/_bad_shape"
        mv -f "${BUF_PATH}" "$(dirname "${BUF_PATH}")/_bad_shape/$(basename "${BUF_PATH}")"
        echo "quarantined stale buffer → _bad_shape/$(basename "${BUF_PATH}")"
      fi
      echo "save-replay: buffer other=${MP2} (reference ego=${REF_EGO}, expect other_n=${EXPECTED_OTHER_AGENTS})"
      run_zeroshot "${REF_EGO}" "${MP2}" "save-replay"
    done
  fi

  for MP1 in "${EGOS[@]}"; do
    for MP2 in "${OTHERS[@]}"; do
      LOG_PATH="${SCENARIO_LOG_DIR}/${MP1}_vs_${MP2}_${eval_one}.json"
      if [[ -f "${LOG_PATH}" ]]; then
        echo "skip ${eval_one}: log exists ${MP1} vs ${MP2} (${LOG_PATH})"
        continue
      fi
      echo "Running ${eval_one}: ${MP1} vs ${MP2}"
      run_zeroshot "${MP1}" "${MP2}" "${eval_one}" --eval.scenario-log-path "${LOG_PATH}"
      echo "Wrote scenario log: ${LOG_PATH}"
    done
  done
}

echo "TRAIN_MODE=${TRAIN_MODE_LIST[*]}  EVAL_MODE=${EVAL_LIST[*]}  UNSEEN=${UNSEEN_LIST[*]}  POP=${POP_NAME}  STRATEGY=${STRATEGY}  MAP=${EVAL_MAP_DIR}"

for tm in "${TRAIN_MODE_LIST[@]}"; do
  if [[ -n "${FOLDER_ARG}" ]]; then
    FOLDER="${FOLDER_ARG}"
  else
    resolve_train_mode "${tm}"
    preferred_folder="${PBT_MODE}_${STRATEGY}_${POP_SHORT}"
    legacy_folder="${TRAIN_MODE_LEGACY}-${STRATEGY}-${POP_NAME}-wandb"
    FOLDER="$(pick_ego_folder "${preferred_folder}" "${legacy_folder}")"
  fi

  for UNSEEN_MODE in "${UNSEEN_LIST[@]}"; do
    for eval_one in "${EVAL_LIST[@]}"; do
      run_one "${eval_one}"
    done
  done
done
