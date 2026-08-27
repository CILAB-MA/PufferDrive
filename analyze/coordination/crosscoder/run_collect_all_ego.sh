#!/usr/bin/env bash
# ============================================================
# Collect activations under all three ego modes in one shot.
#
#   maintain  — primary (a=0,s=0); pipeline/replicate read this
#   record    — ego driven by record_s${DRIVER_SEED} argmax
#   reactive  — ego driven by reactive_s${DRIVER_SEED} argmax
#
# Within each mode every policy still forwards on the same obs.
# Usage:
#   bash analyze/coordination/crosscoder/run_collect_all_ego.sh
#   FORCE=1 bash analyze/coordination/crosscoder/run_collect_all_ego.sh
#   NUM_TRAIN_MAPS=100 NUM_VAL_MAPS=100 bash .../run_collect_all_ego.sh
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COORD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$COORD_DIR/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${COORD_DIR}${PYTHONPATH:+:$PYTHONPATH}"

RESULT_ROOT="${RESULT_ROOT:-/data/puffer/crosscoder/crosscoder_mechanism}"
NUM_TRAIN_MAPS="${NUM_TRAIN_MAPS:-10000}"
NUM_VAL_MAPS="${NUM_VAL_MAPS:-10000}"
SHARD_SIZE="${SHARD_SIZE:-1000}"
DRIVER_SEED="${DRIVER_SEED:-42}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
FORCE="${FORCE:-0}"

have() { [[ -e "$1" ]]; }

echo "========== collect all ego modes =========="
echo "  out=$RESULT_ROOT"
echo "  maps train=$NUM_TRAIN_MAPS val=$NUM_VAL_MAPS shard=$SHARD_SIZE"
echo "  driver_seed=$DRIVER_SEED device=$DEVICE gpu=$GPU_ID force=$FORCE"
echo "  modes: maintain | record | reactive"

force_flag=()
[[ "$FORCE" == "1" ]] && force_flag+=(--force)

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$SCRIPT_DIR/collect.py" \
  --out-root "$RESULT_ROOT" \
  --num-train-maps "$NUM_TRAIN_MAPS" \
  --num-val-maps "$NUM_VAL_MAPS" \
  --shard-size "$SHARD_SIZE" \
  --device "$DEVICE" \
  --ego-mode all \
  --driver-seed "$DRIVER_SEED" \
  "${force_flag[@]}"

base="$RESULT_ROOT/policy_seed_replication"
echo "========== done =========="
echo "  maintain : $base/{train,validation}_dataset/"
echo "  record   : $base/ego_record/{train,validation}_dataset/"
echo "  reactive : $base/ego_reactive/{train,validation}_dataset/"
for path in \
  "$base/validation_dataset/shards" \
  "$base/ego_record/validation_dataset/shards" \
  "$base/ego_reactive/validation_dataset/shards"
do
  if have "$path"; then
    echo "  ok $path"
  else
    echo "  MISSING $path" >&2
  fi
done
