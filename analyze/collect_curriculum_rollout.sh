#!/usr/bin/env bash
# Collect curriculum replay rollouts (save-population + order JSON).
#
# Always uses ORDER_PATH (type_to_population). Staged type schedule over
# NUM_ROLLOUTS: early = easy only, later = mix including harder types.
# Each type folder mixes the first NUM_CKPTS sorted *.pt.
#
# Writes:
#   splits/actions_*.npy, types_*.npy, population_keys_*.npy
#   population_manifest.json
# Merge after collect:
#   python analyze/data_concat.py --population-path "$OUT_PATH" --total-rollouts "$NUM_ROLLOUTS"
#
# Usage:
#   ./analyze/collect_curriculum_rollout.sh 0
#   ./analyze/collect_curriculum_rollout.sh 0 0 25
#
#   # 2-GPU parallel:
#   PARALLEL=1 GPUS=0,1 ./analyze/collect_curriculum_rollout.sh
#
#   # Override defaults:
#   ORDER_PATH=analyze/curriculum_collect_order.example.json \
#   OUT_PATH=/data/puffer/popul_curriculum \
#   NUM_CKPTS=3 NUM_ROLLOUTS=50 \
#     ./analyze/collect_curriculum_rollout.sh 0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORDER_PATH="${ORDER_PATH:-${SCRIPT_DIR}/curriculum_collect_order.example.json}"
OUT_PATH="${OUT_PATH:-/data/puffer/popul_curriculum}"
NUM_CKPTS="${NUM_CKPTS:-4}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-50}"
PARALLEL="${PARALLEL:-0}"
GPUS="${GPUS:-0,1}"
SKIP_SMOKE="${SKIP_SMOKE:-1}"

if [[ "${ORDER_PATH}" == "..." || "${OUT_PATH}" == "..." ]]; then
  echo "ORDER_PATH/OUT_PATH still set to placeholder '...'. Use real paths, e.g.:" >&2
  echo "  ORDER_PATH=analyze/curriculum_collect_order.example.json OUT_PATH=/data/puffer/popul_curriculum" >&2
  exit 1
fi
if [[ ! -f "${ORDER_PATH}" ]]; then
  echo "ORDER_PATH not found: ${ORDER_PATH}" >&2
  exit 1
fi
if [[ "${NUM_CKPTS}" == "0" || -z "${NUM_CKPTS}" ]]; then
  echo "NUM_CKPTS=N is required (first N ckpts mixed per type)" >&2
  exit 1
fi

echo "========== curriculum collect =========="
echo "  order=${ORDER_PATH}"
echo "  out=${OUT_PATH}"
echo "  num_ckpts=${NUM_CKPTS}  num_rollouts=${NUM_ROLLOUTS}"
echo "  parallel=${PARALLEL} gpus=${GPUS} skip_smoke=${SKIP_SMOKE}"
echo "========================================"

build_extra_args() {
  local extra=()
  extra+=(--pbt.collect-order-path "${ORDER_PATH}")
  extra+=(--pbt.population-path "${OUT_PATH}")
  extra+=(--pbt.collect-num-checkpoints "${NUM_CKPTS}")
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
  echo "curriculum collect gpu=${gpu} start=${start} end=${end} total=${total} num_ckpts=${NUM_CKPTS}"
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
  echo "parallel curriculum collect: gpu ${GPU_A}:[0,${MID})  gpu ${GPU_B}:[${MID},${TOTAL})"
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
  echo "  python analyze/data_concat.py --population-path ${OUT_PATH} --total-rollouts ${TOTAL}"
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
