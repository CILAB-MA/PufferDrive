#!/usr/bin/env bash
set -euo pipefail
GPU_ID=${1:-0}
FOLDER=${2:-reactive_nominal} # or replay, selfplay
POPULATION_MODE=${3:-popul_lane} # population folder
MODE=${4:-unseen_other_rewards} # unseen_other_rewards | unseen_other_seeds | ...
# replay | reactive | reactive-play | both
EVAL_MODE=${5:-}
SCENARIO_LOG_DIR=${6:-/data/puffer/results/${FOLDER}/${MODE}/scenario_logs}
mkdir -p "${SCENARIO_LOG_DIR}"
EGOS=()
OTHERS=()

# Default EVAL_MODE from MODE if unset
if [[ -z "${EVAL_MODE}" ]]; then
  if [[ "${MODE}" == "unseen_other_rewards" || "${MODE}" == "unseen_other_seeds" ]]; then
    EVAL_MODE="replay"
  else
    EVAL_MODE="reactive-play"
  fi
fi
case "${EVAL_MODE}" in
  reactive) EVAL_MODE="reactive-play" ;;
  replay|reactive-play|both) ;;
  *)
    echo "EVAL_MODE must be replay | reactive | reactive-play | both (got: ${EVAL_MODE})" >&2
    exit 1
    ;;
esac

for f in /data/puffer/experiments/${FOLDER}/puffer_drive_*.pt; do
  bn=$(basename "$f") # puffer_drive_xxx.pt
  id=${bn#puffer_drive_} # xxx.pt
  id=${id%.pt} # xxx
  EGOS+=("$id")
done

for f in /data/puffer/${POPULATION_MODE}/${MODE}/puffer_drive_*.pt; do
  bn=$(basename "$f") # puffer_drive_xxx.pt
  id=${bn#puffer_drive_} # xxx.pt
  id=${id%.pt} # xxx
  OTHERS+=("$id")
done

echo "Found models: ${EGOS[*]} ${OTHERS[*]}"
echo "EVAL_MODE=${EVAL_MODE}"

run_zeroshot() {
  local ego_id=$1 other_id=$2 zsm=$3
  shift 3
  local -a extra=("$@")
  CUDA_VISIBLE_DEVICES=$GPU_ID puffer zeroshot puffer_drive \
    --load-multiple-model-path \
      "/data/puffer/experiments/${FOLDER}/puffer_drive_${ego_id}.pt" \
      "/data/puffer/${POPULATION_MODE}/${MODE}/puffer_drive_${other_id}.pt" \
    --zero-shot-mode "${zsm}" \
    --env.termination-mode "0" \
    --pbt.pbt-mode "replay" \
    "${extra[@]}"
}

# Must match evaluator.save_replay / play_replay: path[-11:-3] of other ckpt.
other_buffer_path() {
  local other_id=$1
  local other_ckpt="/data/puffer/${POPULATION_MODE}/${MODE}/puffer_drive_${other_id}.pt"
  local buf_id=${other_ckpt: -11:8}
  echo "/data/puffer/experiments/${MODE}/other_action_buffer/other_actions_${buf_id}.npy"
}

need_replay=0
need_reactive=0
case "${EVAL_MODE}" in
  replay) need_replay=1 ;;
  reactive-play) need_reactive=1 ;;
  both) need_replay=1; need_reactive=1 ;;
esac

# save-replay buffers only required for replay (frozen other)
if [[ "${need_replay}" -eq 1 ]]; then
  if [[ ${#EGOS[@]} -eq 0 || ${#OTHERS[@]} -eq 0 ]]; then
    echo "Need at least one ego under experiments/${FOLDER} and one other under ${POPULATION_MODE}/${MODE}." >&2
    exit 1
  fi
  REF_EGO="${EGOS[0]}"
  for MP2 in "${OTHERS[@]}"; do
    BUF_PATH=$(other_buffer_path "${MP2}")
    if [[ -f "${BUF_PATH}" ]]; then
      echo "skip save-replay: buffer exists other=${MP2} (${BUF_PATH})"
      continue
    fi
    echo "save-replay: buffer other=${MP2} (reference ego=${REF_EGO})"
    run_zeroshot "${REF_EGO}" "${MP2}" "save-replay"
  done
fi

ZSMS=()
[[ "${need_replay}" -eq 1 ]] && ZSMS+=("replay")
[[ "${need_reactive}" -eq 1 ]] && ZSMS+=("reactive-play")

for MP1 in "${EGOS[@]}"; do
  for MP2 in "${OTHERS[@]}"; do
    for ZSM in "${ZSMS[@]}"; do
      LOG_PATH="${SCENARIO_LOG_DIR}/${MP1}_vs_${MP2}_${ZSM}.json"
      if [[ -f "${LOG_PATH}" ]]; then
        echo "skip ${ZSM}: log exists ${MP1} vs ${MP2} (${LOG_PATH})"
        continue
      fi
      echo "Running ${ZSM}: ${MP1} vs ${MP2}"
      run_zeroshot "${MP1}" "${MP2}" "${ZSM}" --eval.scenario-log-path "${LOG_PATH}"
      echo "Wrote scenario log: ${LOG_PATH}"
    done
  done
done
