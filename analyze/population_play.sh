#!/usr/bin/env bash
set -euo pipefail
GPU_ID=${1:-0}
POPULATION_MODE=${2:-popul_lane}

EXP_DIR="/data/puffer/experiments/${POPULATION_MODE}"
SELFPLAY_DIR="/data/puffer/${POPULATION_MODE}/selfplay"
RESULTS_JSON="/data/puffer/results/${POPULATION_MODE}/selfplay/zeroshot_reactive.json"
mkdir -p "${EXP_DIR}" "${SELFPLAY_DIR}"

MODELS=()
for f in /data/puffer/${POPULATION_MODE}/puffer_drive_*.pt; do
  [[ -f "$f" ]] || continue
  bn=$(basename "$f")
  id=${bn#puffer_drive_}
  id=${id%.pt}
  MODELS+=("$id")
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "No models found under /data/puffer/${POPULATION_MODE}/." >&2
  exit 1
fi

echo "Population self-play (reactive-play): ${MODELS[*]}"

for MP in "${MODELS[@]}"; do
  MODEL_PATH="/data/puffer/${POPULATION_MODE}/puffer_drive_${MP}.pt"
  EGO_PATH="${EXP_DIR}/puffer_drive_${MP}.pt"
  OTHER_PATH="${SELFPLAY_DIR}/puffer_drive_${MP}.pt"
  ln -sf "${MODEL_PATH}" "${EGO_PATH}"
  ln -sf "${MODEL_PATH}" "${OTHER_PATH}"
  echo "Running reactive-play self-play: ${MP}"
  CUDA_VISIBLE_DEVICES=$GPU_ID puffer zeroshot puffer_drive \
    --load-multiple-model-path "${EGO_PATH}" "${OTHER_PATH}" \
    --zero-shot-mode "reactive-play" --env.termination-mode "0"
done

python3 analyze/aggregate_selfplay.py "${RESULTS_JSON}"
