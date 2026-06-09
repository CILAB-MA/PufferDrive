#!/usr/bin/env bash
GPU_ID=${1:-0}
FOLDER=${2:-reactive_nominal} # or replay, selfplay
POPULATION_MODE=${3:-popul_lane} # population folder
MODE=${4:-unseen_other_rewards} # unseen_other_rewards | unseen_other_seeds | ...
SCENARIO_LOG_DIR=${5:-/data/puffer/results/${FOLDER}/${MODE}/scenario_logs}
mkdir -p "${SCENARIO_LOG_DIR}"
EGOS=()
OTHERS=()

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

# unseen_other_rewards: record each opponent's actions once (save-replay), then main grid uses replay (frozen other).
if [[ "${MODE}" == "unseen_other_rewards" ]]; then
  if [[ ${#EGOS[@]} -eq 0 || ${#OTHERS[@]} -eq 0 ]]; then
    echo "Need at least one ego under experiments/${FOLDER} and one other under ${POPULATION_MODE}/${MODE}." >&2
    exit 1
  fi
  REF_EGO="${EGOS[0]}"
  for MP2 in "${OTHERS[@]}"; do
    echo "save-replay: buffer other=${MP2} (reference ego=${REF_EGO})"
    CUDA_VISIBLE_DEVICES=$GPU_ID puffer zeroshot puffer_drive \
      --load-multiple-model-path "/data/puffer/experiments/${FOLDER}/puffer_drive_${REF_EGO}.pt" \
                                 "/data/puffer/${POPULATION_MODE}/${MODE}/puffer_drive_${MP2}.pt" \
      --zero-shot-mode "save-replay" --env.termination-mode "0"
  done
fi

for MP1 in "${EGOS[@]}"; do
  for MP2 in "${OTHERS[@]}"; do
    if [[ "${MODE}" == "unseen_other_rewards" ]]; then
      ZSM="replay"
      echo "Running replay: ${MP1} vs ${MP2}"
    else
      ZSM="reactive-play"
      echo "Running reactive-play: ${MP1} vs ${MP2}"
    fi
    LOG_PATH="${SCENARIO_LOG_DIR}/${MP1}_vs_${MP2}_${ZSM}.json"
    CUDA_VISIBLE_DEVICES=$GPU_ID puffer zeroshot puffer_drive \
      --load-multiple-model-path "/data/puffer/experiments/${FOLDER}/puffer_drive_${MP1}.pt" \
                                 "/data/puffer/${POPULATION_MODE}/${MODE}/puffer_drive_${MP2}.pt" \
      --zero-shot-mode "${ZSM}" --env.termination-mode "0" \
      --eval.scenario-log-path "${LOG_PATH}"
    echo "Wrote scenario log: ${LOG_PATH}"
  done
done
