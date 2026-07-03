#!/usr/bin/env bash
# Population / experiment-dir human-replay eval: each checkpoint vs human partners.
# Writes aggregate metrics to logreplay.json and per-map scenario logs when enabled.
#
# Usage:
#   # Legacy: population under /data/puffer/<POPULATION_MODE>/
#   ./analyze/population_selfplay_human_eval.sh [GPU_ID] [POPULATION_MODE]
#
#   # Checkpoints already under experiments/ (e.g. selfplay training run)
#   CKPT_DIR=/data/puffer/experiments/selfplay ./analyze/population_selfplay_human_eval.sh 0
#
#   # Disable per-map scenario logs (aggregate only)
#   SCENARIO_LOG_DIR= ./analyze/population_selfplay_human_eval.sh 0
#
#   NUM_MAPS=10000 ./analyze/population_selfplay_human_eval.sh 0 replay_0.25
#
#   # CPU-only evaluation (no GPU required): pass "cpu" as GPU_ID
#   ./analyze/population_selfplay_human_eval.sh cpu

set -euo pipefail
GPU_ID=${1:-0}
POPULATION_MODE=${2:-popul_lane}
NUM_MAPS=${NUM_MAPS:-10000}

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

# Results path follows puffer eval path parsing: .../experiments/<exp>/... -> exp name.
if [[ "$CKPT_DIR" == */experiments/* ]]; then
  EXP_NAME="$(basename "$CKPT_DIR")"
else
  EXP_NAME="${POPULATION_MODE}"
fi

RESULTS_JSON="/data/puffer/results/${EXP_NAME}/logreplay.json"
# Set SCENARIO_LOG_DIR= to skip per-map logs.
if [[ -z "${SCENARIO_LOG_DIR+x}" ]]; then
  SCENARIO_LOG_DIR="/data/puffer/results/${EXP_NAME}/human_replay/scenario_logs"
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

echo "Human-replay eval from ${CKPT_DIR}"
echo "  models: ${MODELS[*]}"
echo "  num_maps: ${NUM_MAPS}"
echo "  aggregate: ${RESULTS_JSON}"
if [[ -n "$SCENARIO_LOG_DIR" ]]; then
  echo "  scenario logs: ${SCENARIO_LOG_DIR}/<id>_human_replay.json"
fi

for MP in "${MODELS[@]}"; do
  MODEL_PATH="${CKPT_DIR}/puffer_drive_${MP}.pt"

  EXTRA_ARGS=()
  if [[ -n "$SCENARIO_LOG_DIR" ]]; then
    LOG_PATH="${SCENARIO_LOG_DIR}/${MP}_human_replay.json"
    EXTRA_ARGS+=(--eval.scenario-log-path "${LOG_PATH}")
  fi

  echo "Running human-replay eval: ${MP}"
  if [[ "$DEVICE" == "cpu" ]]; then
    CUDA_VISIBLE_DEVICES="" puffer eval puffer_drive \
      --eval.human-replay-eval True \
      --eval.human-replay-save-results True \
      --env.termination-mode "0" \
      --eval.wosac-num-maps "${NUM_MAPS}" \
      --load-model-path "${MODEL_PATH}" \
      --train.device cpu \
      "${EXTRA_ARGS[@]}"
  else
    CUDA_VISIBLE_DEVICES=$GPU_ID puffer eval puffer_drive \
      --eval.human-replay-eval True \
      --eval.human-replay-save-results True \
      --env.termination-mode "0" \
      --eval.wosac-num-maps "${NUM_MAPS}" \
      --load-model-path "${MODEL_PATH}" \
      "${EXTRA_ARGS[@]}"
  fi
  if [[ -n "$SCENARIO_LOG_DIR" ]]; then
    echo "  scenario log: ${LOG_PATH}"
  fi
done

python3 analyze/aggregate_selfplay.py "${RESULTS_JSON}"