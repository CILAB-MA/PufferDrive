#!/usr/bin/env bash
# Fair Record collect bench: train-path policy SPS + storage I/O.
# Writes:
#   /data/puffer/experiments/collect_bench_{pop_short}/collect_summary.json
# Optionally builds amortized Record vs Reactive table if SPS summaries exist.
#
# Usage:
#   ./scripts/bench_replay_collect.sh 0
#   POP_PATH=/data/puffer/popul_lane_nominal ./scripts/bench_replay_collect.sh 0
#   FLUSH_MAX_BYTES=2000000000 ./scripts/bench_replay_collect.sh 0   # ~2GB smoke flush
#   SKIP_FAIR_TABLE=1 ./scripts/bench_replay_collect.sh 0
#
# Env:
#   POP_PATH / POPULATION_PATH   default /data/puffer/popul_lane_nominal
#   TOTAL_TIMESTEPS             default 20000000 (env/policy phase)
#   N_COMBOS                    default 10 (rollouts for T_collect estimate)
#   WARMUP_STEPS                default 100000
#   SEED                        default 0
#   FLUSH_MAX_BYTES             0 = write sized to N_COMBOS
#   KEEP_FLUSH_FILE             0/1
#   STORAGE_COPY_DEST           optional copy target for t_storage_copy
#   RECORD_SPS_SUMMARY          default sps_uniform_record/sps_summary.json
#   REACTIVE_SPS_SUMMARY        default sps_uniform_reactive/sps_summary.json
#   FAIR_N                      default 100000000
#   FAIR_M                      default 4
#   SKIP_ENV_POLICY / SKIP_FLUSH / SKIP_FAIR_TABLE
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-0}"
POP_PATH="${POP_PATH:-${POPULATION_PATH:-/data/puffer/popul_lane_nominal}}"
POP_PATH="${POP_PATH%/}"
POP_NAME="$(basename "${POP_PATH}")"
POP_SHORT="${POP_NAME#popul_}"

OUT_DIR="${OUT_DIR:-/data/puffer/experiments/collect_bench_${POP_SHORT}}"
SEED="${SEED:-0}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20000000}"
WARMUP_STEPS="${WARMUP_STEPS:-100000}"
N_COMBOS="${N_COMBOS:-10}"
FLUSH_MAX_BYTES="${FLUSH_MAX_BYTES:-0}"
KEEP_FLUSH_FILE="${KEEP_FLUSH_FILE:-0}"
STORAGE_COPY_DEST="${STORAGE_COPY_DEST:-}"
SKIP_ENV_POLICY="${SKIP_ENV_POLICY:-0}"
SKIP_FLUSH="${SKIP_FLUSH:-0}"
SKIP_FAIR_TABLE="${SKIP_FAIR_TABLE:-0}"
FAIR_N="${FAIR_N:-100000000}"
FAIR_M="${FAIR_M:-4}"
RECORD_SPS_SUMMARY="${RECORD_SPS_SUMMARY:-/data/puffer/experiments/sps_uniform_record/sps_summary.json}"
REACTIVE_SPS_SUMMARY="${REACTIVE_SPS_SUMMARY:-/data/puffer/experiments/sps_uniform_reactive/sps_summary.json}"

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  echo "python/python3 not found" >&2
  exit 1
fi

if [[ ! -d "${POP_PATH}" ]]; then
  echo "Population not found: ${POP_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

CMD=(
  "${PYTHON}" "${ROOT}/analyze/bench_replay_collect.py"
  --population-path "${POP_PATH}"
  --out-dir "${OUT_DIR}"
  --seed "${SEED}"
  --total-timesteps "${TOTAL_TIMESTEPS}"
  --warmup-steps "${WARMUP_STEPS}"
  --n-combos "${N_COMBOS}"
)

if [[ "${FLUSH_MAX_BYTES}" != "0" ]]; then
  CMD+=(--flush-max-bytes "${FLUSH_MAX_BYTES}")
fi
if [[ "${KEEP_FLUSH_FILE}" == "1" || "${KEEP_FLUSH_FILE}" == "true" ]]; then
  CMD+=(--keep-flush-file)
fi
if [[ -n "${STORAGE_COPY_DEST}" ]]; then
  CMD+=(--storage-copy-dest "${STORAGE_COPY_DEST}")
fi
if [[ "${SKIP_ENV_POLICY}" == "1" || "${SKIP_ENV_POLICY}" == "true" ]]; then
  CMD+=(--skip-env-policy)
  if [[ -n "${ENV_POLICY_SUMMARY:-}" ]]; then
    CMD+=(--env-policy-summary "${ENV_POLICY_SUMMARY}")
  else
    echo "SKIP_ENV_POLICY=1 requires ENV_POLICY_SUMMARY=..." >&2
    exit 1
  fi
fi
if [[ "${SKIP_FLUSH}" == "1" || "${SKIP_FLUSH}" == "true" ]]; then
  CMD+=(--skip-flush)
fi

echo "========== bench_replay_collect =========="
echo "  GPU=${GPU_ID}"
echo "  population=${POP_PATH}"
echo "  out_dir=${OUT_DIR}"
echo "  n_combos=${N_COMBOS}"
echo "  total_timesteps=${TOTAL_TIMESTEPS}"
echo "  flush_max_bytes=${FLUSH_MAX_BYTES}"
echo "=========================================="

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"

SUMMARY="${OUT_DIR}/collect_summary.json"
if [[ ! -f "${SUMMARY}" ]]; then
  echo "Missing ${SUMMARY}" >&2
  exit 1
fi

if [[ "${SKIP_FAIR_TABLE}" != "1" && "${SKIP_FAIR_TABLE}" != "true" ]]; then
  if [[ -f "${RECORD_SPS_SUMMARY}" && -f "${REACTIVE_SPS_SUMMARY}" ]]; then
    echo "Building fair SPS table ..."
    "${PYTHON}" "${ROOT}/analyze/fair_sps_compare.py" \
      --record-sps "${RECORD_SPS_SUMMARY}" \
      --reactive-sps "${REACTIVE_SPS_SUMMARY}" \
      --collect "${SUMMARY}" \
      --N "${FAIR_N}" \
      --M "${FAIR_M}" \
      --out "${OUT_DIR}/fair_sps_table.json"
  else
    echo "Skip fair table (missing SPS summaries):" >&2
    echo "  record=${RECORD_SPS_SUMMARY}" >&2
    echo "  reactive=${REACTIVE_SPS_SUMMARY}" >&2
  fi
fi

echo ""
echo "Done."
echo "  collect: ${SUMMARY}"
echo "  fair:    ${OUT_DIR}/fair_sps_table.json"
