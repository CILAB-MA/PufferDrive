#!/usr/bin/env bash
# Collect replay rollouts (save-population), optionally with a curriculum order JSON.
#
# Fast path (default):
#   - env reused across rollouts (resample_frequency=0 during collect)
#   - smoke-test skipped (skip_collect_smoke_test=True)
#   - optional 2-GPU parallel shards via PARALLEL=1
#
# Curriculum mode (ORDER_PATH + NUM_CKPTS):
#   - each type folder uses the first NUM_CKPTS sorted *.pt (mixed within the type)
#   - NUM_ROLLOUTS total rollouts with a staged type schedule
#
# Also writes per-agent population keys (which ckpt acted for each agent):
#   splits/population_keys_*.npy → saved/population_keys.npy
#   population_manifest.json (lookup: key -> population path + checkpoint)
#
# Usage:
#   ORDER_PATH=analyze/curriculum_collect_order.example.json \
#   OUT_PATH=/data/puffer/popul_curriculum \
#   NUM_CKPTS=3 NUM_ROLLOUTS=50 \
#     ./analyze/collect_replay_rollout.sh 0
#
#   # 2-GPU parallel:
#   ORDER_PATH=analyze/curriculum_collect_order.example.json \
#   OUT_PATH=/data/puffer/popul_curriculum \
#   NUM_CKPTS=3 NUM_ROLLOUTS=50 PARALLEL=1 GPUS=0,1 \
#     ./analyze/collect_replay_rollout.sh
#
#   # Optional single-GPU shard:
#   ORDER_PATH=analyze/curriculum_collect_order.example.json \
#   OUT_PATH=/data/puffer/popul_curriculum NUM_CKPTS=3 NUM_ROLLOUTS=50 \
#     ./analyze/collect_replay_rollout.sh 0 0 25
set -euo pipefail

ORDER_PATH="${ORDER_PATH:-}"
OUT_PATH="${OUT_PATH:-}"
NUM_CKPTS="${NUM_CKPTS:-0}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-50}"
PARALLEL="${PARALLEL:-0}"
GPUS="${GPUS:-0,1}"
SKIP_SMOKE="${SKIP_SMOKE:-1}"

if [[ "${ORDER_PATH}" == "..." || "${OUT_PATH}" == "..." ]]; then
  echo "ORDER_PATH/OUT_PATH still set to placeholder '...'. Use real paths, e.g.:" >&2
  echo "  ORDER_PATH=analyze/curriculum_collect_order.example.json OUT_PATH=/data/puffer/popul_curriculum" >&2
  exit 1
fi

build_extra_args() {
  local extra=()
  if [[ -n "${ORDER_PATH}" ]]; then
    if [[ -z "${OUT_PATH}" ]]; then
      echo "OUT_PATH is required when ORDER_PATH is set (output population root)" >&2
      exit 1
    fi
    if [[ "${NUM_CKPTS}" == "0" || -z "${NUM_CKPTS}" ]]; then
      echo "NUM_CKPTS=N is required with ORDER_PATH (first N ckpts mixed per type)" >&2
      exit 1
    fi
    extra+=(--pbt.collect-order-path "${ORDER_PATH}")
    extra+=(--pbt.population-path "${OUT_PATH}")
    extra+=(--pbt.collect-num-checkpoints "${NUM_CKPTS}")
  else
    if [[ -n "${NUM_CKPTS}" && "${NUM_CKPTS}" != "0" ]]; then
      extra+=(--pbt.collect-num-checkpoints "${NUM_CKPTS}")
    fi
    if [[ -n "${OUT_PATH}" ]]; then
      extra+=(--pbt.population-path "${OUT_PATH}")
    fi
  fi
  if [[ "${SKIP_SMOKE}" == "1" || "${SKIP_SMOKE}" == "True" || "${SKIP_SMOKE}" == "true" ]]; then
    extra+=(--pbt.skip-collect-smoke-test True)
  else
    extra+=(--pbt.skip-collect-smoke-test False)
  fi
  printf '%s\n' "${extra[@]}"
}

run_shard() {
  local gpu="$1"
  local start="$2"
  local end="$3"
  local total="$4"
  mapfile -t EXTRA_ARGS < <(build_extra_args)
  echo "collect gpu=${gpu} start=${start} end=${end} total=${total} num_ckpts=${NUM_CKPTS:-0}"
  CUDA_VISIBLE_DEVICES="${gpu}" puffer zeroshot puffer_drive \
    --pbt.num-collect-rollout="${total}" \
    --pbt.collect-start-idx="${start}" \
    --pbt.collect-end-idx="${end}" \
    "${EXTRA_ARGS[@]}"
}

TOTAL="${NUM_ROLLOUTS}"

if [[ "${PARALLEL}" == "1" ]]; then
  IFS=',' read -r GPU_A GPU_B <<< "${GPUS}"
  if [[ -z "${GPU_A}" || -z "${GPU_B}" ]]; then
    echo "PARALLEL=1 requires GPUS=a,b (got GPUS=${GPUS})" >&2
    exit 1
  fi
  MID=$((TOTAL / 2))
  if [[ "${MID}" -lt 1 || "${MID}" -ge "${TOTAL}" ]]; then
    echo "NUM_ROLLOUTS=${TOTAL} too small to split across 2 GPUs" >&2
    exit 1
  fi
  echo "parallel collect: gpu ${GPU_A}:[0,${MID})  gpu ${GPU_B}:[${MID},${TOTAL})"
  run_shard "${GPU_A}" 0 "${MID}" "${TOTAL}" &
  PID_A=$!
  run_shard "${GPU_B}" "${MID}" "${TOTAL}" "${TOTAL}" &
  PID_B=$!
  status=0
  wait "${PID_A}" || status=$?
  wait "${PID_B}" || status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "One or more parallel shards failed (status=${status})" >&2
    exit "${status}"
  fi
  echo "Both shards done. Merge with:"
  echo "  python analyze/data_concat.py --population-path ${OUT_PATH:-<population_path>} --total-rollouts ${TOTAL}"
  exit 0
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <cuda_device> [collect_start_idx] [collect_end_idx]" >&2
  echo "  ORDER_PATH=... OUT_PATH=... NUM_CKPTS=N [NUM_ROLLOUTS=50]" >&2
  echo "  PARALLEL=1 GPUS=0,1 ...   # split rollouts across 2 GPUs" >&2
  exit 1
fi

START="${2:-0}"
END="${3:-${TOTAL}}"
run_shard "${1}" "${START}" "${END}" "${TOTAL}"
