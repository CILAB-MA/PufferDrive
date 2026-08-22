#!/usr/bin/env bash
# ============================================================
# Train Crosscoder on all three collect ego modes.
#
#   maintain  → $RESULT_10K/summary.json          (primary)
#   record    → ${RESULT_10K}_ego_record/
#   reactive  → ${RESULT_10K}_ego_reactive/
#
# Requires collect shards from run_collect_all_ego.sh (or collect.py --ego-mode all).
# Usage:
#   bash analyze/coordination/crosscoder/run_train_crosscoder_all.sh
#   FORCE=1 bash analyze/coordination/crosscoder/run_train_crosscoder_all.sh
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
RESULT_10K="${RESULT_10K:-/data/puffer/crosscoder/crosscoder_10k}"
ACTS_ROOT="${ACTS_ROOT:-$RESULT_ROOT/policy_seed_replication}"
REC_KEY="${REC_KEY:-record_s3}"
REA_KEY="${REA_KEY:-reactive_s11}"
EPOCHS="${EPOCHS:-25}"
SEEDS="${SEEDS:-1}"
GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
FORCE="${FORCE:-0}"

have() { [[ -e "$1" ]]; }

modes=(maintain record reactive)

echo "========== train Crosscoder all ego modes =========="
echo "  acts=$ACTS_ROOT"
echo "  pair=$REC_KEY vs $REA_KEY"
echo "  epochs=$EPOCHS seeds=$SEEDS device=$DEVICE gpu=$GPU_ID force=$FORCE"

for mode in "${modes[@]}"; do
  if [[ "$mode" == "maintain" ]]; then
    acts_dir="$ACTS_ROOT/train_dataset/shards"
    out="$RESULT_10K"
  else
    acts_dir="$ACTS_ROOT/ego_${mode}/train_dataset/shards"
    out="${RESULT_10K}_ego_${mode}"
  fi

  echo ""
  echo "----- ego_mode=$mode → $out -----"

  if ! have "$acts_dir"; then
    echo "MISSING collect for $mode ($acts_dir). Run run_collect_all_ego.sh first." >&2
    exit 1
  fi
  if [[ "$FORCE" != "1" ]] && have "$out/summary.json"; then
    echo "[reuse] $out/summary.json (set FORCE=1 to regenerate)"
    continue
  fi

  overwrite_flag=()
  [[ "$FORCE" == "1" ]] && overwrite_flag+=(--overwrite)

  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$SCRIPT_DIR/pipeline.py" \
    --out-root "$out" \
    --acts-root "$ACTS_ROOT" \
    --ego-mode "$mode" \
    --rec-key "$REC_KEY" \
    --rea-key "$REA_KEY" \
    --epochs "$EPOCHS" \
    --seeds "$SEEDS" \
    --device "$DEVICE" \
    "${overwrite_flag[@]}"
done

echo ""
echo "========== done =========="
for mode in "${modes[@]}"; do
  if [[ "$mode" == "maintain" ]]; then
    sum="$RESULT_10K/summary.json"
  else
    sum="${RESULT_10K}_ego_${mode}/summary.json"
  fi
  if have "$sum"; then
    echo "  ok $sum"
  else
    echo "  MISSING $sum" >&2
  fi
done
