#!/usr/bin/env bash
# Shared-obs SAE activation collect (no labels).
#
# One human-replay rollout defines shared obs; then encode the same rows with
# selfplay / reactive_0.25 / replay_0.25. Writes activations.npz only
# (no future_*.npz / LP labels on disk).
#
# Usage:
#   TRAIN_NUM_MAPS=10000 VAL_NUM_MAPS=1500 GPU_ID=0 PROBE_STEP=1908 \
#     MAX_MIN_DIST_M=10 ./analyze/sae/run_onpolicy_sae_pipeline.sh
#   SKIP_VAL_COLLECT=1 TRAIN_NUM_MAPS=1000 ./analyze/sae/run_onpolicy_sae_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"

BASE_PATH="${BASE_PATH:-/data/puffer/experiments}"
REFERENCE_EXP="${REFERENCE_EXP:-replay_0.25}"
EXPERIMENTS="${EXPERIMENTS:-selfplay,reactive_0.25,replay_0.25}"
# collect writes under <SAE_ROOT>/human_replay/{training,validation}/
SAE_ROOT="${SAE_ROOT:-/data/puffer/sae}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
TRAIN_NUM_MAPS="${TRAIN_NUM_MAPS:-10000}"
VAL_NUM_MAPS="${VAL_NUM_MAPS:-1500}"
PROBE_STEP="${PROBE_STEP:-1908}"
MAX_MIN_DIST_M="${MAX_MIN_DIST_M:-25.0}"
MAX_TIMESTEPS_PER_PAIR="${MAX_TIMESTEPS_PER_PAIR:-4}"
MIN_TIMESTEP_GAP="${MIN_TIMESTEP_GAP:-8}"
MAX_SAMPLES_PER_SCENE="${MAX_SAMPLES_PER_SCENE:-64}"
SAVE_OBS="${SAVE_OBS:-0}"
FORCE_COLLECT="${FORCE_COLLECT:-0}"
SKIP_TRAIN_COLLECT="${SKIP_TRAIN_COLLECT:-0}"
SKIP_VAL_COLLECT="${SKIP_VAL_COLLECT:-0}"
RUN_DIR="${RUN_DIR:-}"

collect_one() {
  local data_mode="$1"
  local num_maps="$2"

  echo
  echo "################################################################"
  echo "# shared-obs SAE  DATA_MODE=${data_mode}  NUM_MAPS=${num_maps}"
  echo "# reference=${REFERENCE_EXP}  experiments=${EXPERIMENTS}"
  echo "# diversity: max_per_pair=${MAX_TIMESTEPS_PER_PAIR} gap=${MIN_TIMESTEP_GAP} max_per_scene=${MAX_SAMPLES_PER_SCENE}"
  echo "################################################################"

  local args=(
    --base-path "$BASE_PATH"
    --reference-exp "$REFERENCE_EXP"
    --experiments "$EXPERIMENTS"
    --output-dir "$SAE_ROOT"
    --num-maps "$num_maps"
    --device "$DEVICE"
    --data-mode "$data_mode"
    --max-min-dist-m "$MAX_MIN_DIST_M"
    --max-timesteps-per-pair "$MAX_TIMESTEPS_PER_PAIR"
    --min-timestep-gap "$MIN_TIMESTEP_GAP"
    --max-samples-per-scene "$MAX_SAMPLES_PER_SCENE"
  )
  if [[ -n "$PROBE_STEP" ]]; then
    args+=(--probe-step "$PROBE_STEP")
  fi
  if [[ -n "$RUN_DIR" ]]; then
    args+=(--run-dir "$RUN_DIR")
  fi
  if [[ "$FORCE_COLLECT" == "1" ]]; then
    args+=(--force-collect)
  fi
  if [[ "$SAVE_OBS" == "1" ]]; then
    args+=(--save-obs)
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$SCRIPT_DIR/collect_sae_activations.py" "${args[@]}"
}

cd "$REPO_ROOT"
mkdir -p "$SAE_ROOT"

echo "========== List checkpoints =========="
IFS=',' read -ra EXP_LIST <<< "$EXPERIMENTS"
for EXP in "${EXP_LIST[@]}"; do
  EXP="$(echo "$EXP" | xargs)"
  [[ -z "$EXP" ]] && continue
  echo "--- $EXP ---"
  "$PYTHON" "$SCRIPT_DIR/load_ckpt.py" --base-path "$BASE_PATH" --exp-name "$EXP"
done

if [[ "$SKIP_TRAIN_COLLECT" != "1" ]]; then
  collect_one training "$TRAIN_NUM_MAPS"
else
  echo "SKIP_TRAIN_COLLECT=1"
fi

if [[ "$SKIP_VAL_COLLECT" != "1" ]]; then
  collect_one validation "$VAL_NUM_MAPS"
else
  echo "SKIP_VAL_COLLECT=1"
fi

echo
echo "========== Verify =========="
"$PYTHON" <<VERIFY_EOF
import sys
sys.path.insert(0, "$SCRIPT_DIR")
from collect_sae_activations import verify_sae_root
from sae_rollout import parse_csv_list

exps = parse_csv_list("$EXPERIMENTS")
errors = []
for mode in ("training", "validation"):
    if mode == "training" and "$SKIP_TRAIN_COLLECT" == "1":
        continue
    if mode == "validation" and "$SKIP_VAL_COLLECT" == "1":
        continue
    errors.extend(verify_sae_root("$SAE_ROOT", mode, experiments=exps))
if errors:
    for e in errors:
        print("  error:", e)
    raise SystemExit(1)
print("  OK: activations.npz with", ", ".join(f"activation__{e}" for e in exps))
VERIFY_EOF

echo
echo "Shared-obs SAE collect done."
echo "  root: ${SAE_ROOT}/human_replay/{training,validation}/step_$(printf '%06d' "${PROBE_STEP:-0}")/"
echo "  file: activations.npz  (no labels)"
echo "  keys: activation__selfplay, activation__reactive_0.25, activation__replay_0.25"
