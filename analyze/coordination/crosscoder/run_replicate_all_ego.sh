#!/usr/bin/env bash
# ============================================================
# Replicate + subspace on all three collect ego modes.
#
#   maintain  → $RESULT_ROOT/summary.json          (primary)
#   record    → ${RESULT_ROOT}_ego_record/
#   reactive  → ${RESULT_ROOT}_ego_reactive/
#
# Reads collect shards from:
#   maintain : $ACTS_ROOT/{train,validation}_dataset/
#   record   : $ACTS_ROOT/ego_record/...
#   reactive : $ACTS_ROOT/ego_reactive/...
#
# Requires: run_collect_all_ego.sh (or collect.py --ego-mode all)
#           and train_dev_split.json under crosscoder_10k (for subspace).
#
# Usage:
#   bash analyze/coordination/crosscoder/run_replicate_all_ego.sh
#   FORCE=1 bash analyze/coordination/crosscoder/run_replicate_all_ego.sh
#   EGO_MODES=maintain bash .../run_replicate_all_ego.sh
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
ACTS_ROOT="${ACTS_ROOT:-$RESULT_ROOT/policy_seed_replication}"
EPOCHS="${EPOCHS:-20}"
CC_SEEDS="${CC_SEEDS:-5}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
FORCE="${FORCE:-0}"

# space-separated; default all three
if [[ -n "${EGO_MODES:-}" ]]; then
  # shellcheck disable=SC2206
  modes=($EGO_MODES)
else
  modes=(maintain record reactive)
fi

have() { [[ -e "$1" ]]; }

out_for() {
  local mode="$1"
  if [[ "$mode" == "maintain" ]]; then
    echo "$RESULT_ROOT"
  else
    echo "${RESULT_ROOT}_ego_${mode}"
  fi
}

acts_for() {
  local mode="$1"
  if [[ "$mode" == "maintain" ]]; then
    echo "$ACTS_ROOT"
  else
    echo "$ACTS_ROOT/ego_${mode}"
  fi
}

echo "========== replicate all ego modes =========="
echo "  acts=$ACTS_ROOT"
echo "  out primary=$RESULT_ROOT"
echo "  epochs=$EPOCHS cc_seeds=$CC_SEEDS device=$DEVICE gpu=$GPU_ID force=$FORCE"
echo "  modes: ${modes[*]}"

for mode in "${modes[@]}"; do
  acts="$(acts_for "$mode")"
  out="$(out_for "$mode")"
  shards="$acts/train_dataset/shards"

  echo ""
  echo "----- ego_mode=$mode -----"
  echo "  acts=$acts"
  echo "  out=$out"

  if ! have "$shards"; then
    echo "MISSING collect for $mode ($shards). Run run_collect_all_ego.sh first." >&2
    exit 1
  fi
  if [[ "$FORCE" != "1" ]] && have "$out/summary.json" && have "$out/subspace/summary.json"; then
    echo "[reuse] $out/summary.json (set FORCE=1 to regenerate)"
    continue
  fi

  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$SCRIPT_DIR/replicate.py" \
    --out-root "$RESULT_ROOT" \
    --acts-root "$ACTS_ROOT" \
    --ego-mode "$mode" \
    --epochs "$EPOCHS" \
    --cc-seeds "$CC_SEEDS" \
    --device "$DEVICE"
done

echo ""
echo "========== done =========="
for mode in "${modes[@]}"; do
  sum="$(out_for "$mode")/summary.json"
  if have "$sum"; then
    echo "  ok $sum"
  else
    echo "  MISSING $sum" >&2
  fi
done
