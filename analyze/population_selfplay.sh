#!/usr/bin/env bash
# Population / experiment-dir self-play (reactive-play): each checkpoint vs itself.
# Writes aggregate metrics to zeroshot_reactive.json and per-map scenario logs when enabled.
#
# Usage:
#   # Legacy: population under /data/puffer/<POPULATION_MODE>/
#   ./analyze/population_play.sh [GPU_ID] [POPULATION_MODE]
#
#   # Checkpoints already under experiments/ (e.g. selfplay training run)
#   CKPT_DIR=/data/puffer/experiments/selfplay ./analyze/population_play.sh 0
#
#   # Disable per-map scenario logs (aggregate only)
#   SCENARIO_LOG_DIR= ./analyze/population_play.sh 0
#
#   # CPU-only evaluation (no GPU required): pass "cpu" as GPU_ID
#   ./analyze/population_play.sh cpu

set -euo pipefail
GPU_ID=${1:-0}
POPULATION_MODE=${2:-popul_lane}

if [[ "$GPU_ID" == "cpu" ]]; then
  DEVICE="cpu"
else
  DEVICE="cuda"
fi

# Checkpoint directory: explicit CKPT_DIR, else population folder.
if [[ -n "${CKPT_DIR:-}" ]]; then
  CKPT_DIR="$(cd "$CKPT_DIR" && pwd)"
elif [[ -d "/data/puffer/experiments/${POPULATION_MODE}" ]] \
  && compgen -G "/data/puffer/experiments/${POPULATION_MODE}/puffer_drive_*.pt" >/dev/null; then
  CKPT_DIR="/data/puffer/experiments/${POPULATION_MODE}"
else
  CKPT_DIR="/data/puffer/${POPULATION_MODE}"
fi

# Results path follows puffer zeroshot path parsing: .../experiments/<exp>/... -> exp name.
if [[ "$CKPT_DIR" == */experiments/* ]]; then
  EXP_NAME="$(basename "$CKPT_DIR")"
  USE_SYMLINKS=0
else
  EXP_NAME="${POPULATION_MODE}"
  USE_SYMLINKS=1
  EXP_DIR="/data/puffer/experiments/${POPULATION_MODE}"
  SELFPLAY_DIR="/data/puffer/${POPULATION_MODE}/selfplay"
  mkdir -p "${EXP_DIR}" "${SELFPLAY_DIR}"
fi

RESULTS_JSON="/data/puffer/results/${EXP_NAME}/selfplay/zeroshot_reactive.json"
# Set SCENARIO_LOG_DIR= to skip per-map logs.
if [[ -z "${SCENARIO_LOG_DIR+x}" ]]; then
  SCENARIO_LOG_DIR="/data/puffer/results/${EXP_NAME}/selfplay/scenario_logs"
fi
mkdir -p "$(dirname "$RESULTS_JSON")"
if [[ -n "$SCENARIO_LOG_DIR" ]]; then
  mkdir -p "$SCENARIO_LOG_DIR"
fi

MODELS=()
for f in "${CKPT_DIR}"/puffer_drive_*.pt; do
  [[ -f "$f" ]] || continue
  bn=$(basename "$f")
  id=${bn#puffer_drive_}
  id=${id%.pt}
  MODELS+=("$id")
done

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "No models found under ${CKPT_DIR}/." >&2
  exit 1
fi

echo "Self-play (reactive-play) from ${CKPT_DIR}"
echo "  models: ${MODELS[*]}"
echo "  aggregate: ${RESULTS_JSON}"
if [[ -n "$SCENARIO_LOG_DIR" ]]; then
  echo "  scenario logs: ${SCENARIO_LOG_DIR}/<id>_selfplay.json"
fi

for MP in "${MODELS[@]}"; do
  MODEL_PATH="${CKPT_DIR}/puffer_drive_${MP}.pt"
  if [[ "$USE_SYMLINKS" -eq 1 ]]; then
    EGO_PATH="${EXP_DIR}/puffer_drive_${MP}.pt"
    OTHER_PATH="${SELFPLAY_DIR}/puffer_drive_${MP}.pt"
    ln -sf "${MODEL_PATH}" "${EGO_PATH}"
    ln -sf "${MODEL_PATH}" "${OTHER_PATH}"
  else
    # Same file twice: exp/mode both resolve to CKPT_DIR basename (e.g. selfplay).
    EGO_PATH="${MODEL_PATH}"
    OTHER_PATH="${MODEL_PATH}"
  fi

  EXTRA_ARGS=()
  if [[ -n "$SCENARIO_LOG_DIR" ]]; then
    LOG_PATH="${SCENARIO_LOG_DIR}/${MP}_selfplay.json"
    EXTRA_ARGS+=(--eval.scenario-log-path "${LOG_PATH}")
  fi

  echo "Running reactive-play self-play: ${MP}"
  if [[ "$DEVICE" == "cpu" ]]; then
    CUDA_VISIBLE_DEVICES="" puffer zeroshot puffer_drive \
      --load-multiple-model-path "${EGO_PATH}" "${OTHER_PATH}" \
      --zero-shot-mode "reactive-play" --env.termination-mode "0" \
      --train.device cpu \
      "${EXTRA_ARGS[@]}"
  else
    CUDA_VISIBLE_DEVICES=$GPU_ID puffer zeroshot puffer_drive \
      --load-multiple-model-path "${EGO_PATH}" "${OTHER_PATH}" \
      --zero-shot-mode "reactive-play" --env.termination-mode "0" \
      "${EXTRA_ARGS[@]}"
  fi
  if [[ -n "$SCENARIO_LOG_DIR" ]]; then
    echo "  scenario log: ${LOG_PATH}"
  fi
done

python3 analyze/aggregate_selfplay.py "${RESULTS_JSON}"