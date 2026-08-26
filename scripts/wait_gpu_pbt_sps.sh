#!/usr/bin/env bash
# Wait until a GPU is idle, then run scripts/pbt_sps_seed.sh.
#
# Usage:
#   MODE=reactive STRATEGY=uniform ./scripts/wait_gpu_pbt_sps.sh 0
#   MODE=record STRATEGY=uniform ./scripts/wait_gpu_pbt_sps.sh 0
#   GPU_ID=1 MODE=reactive ./scripts/wait_gpu_pbt_sps.sh   # same; arg wins
#
# Extra args after GPU_ID are forwarded to pbt_sps_seed.sh (e.g. seed list):
#   MODE=reactive ./scripts/wait_gpu_pbt_sps.sh 0 "42 3"
#
# Env:
#   MEM_MIB_MAX   idle if used memory <= this (default 512)
#   UTIL_MAX      idle if GPU util % <= this (default 5)
#   IDLE_CHECKS   consecutive idle polls required (default 2)
#   POLL_SEC      seconds between polls (default 30)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPS_SH="${ROOT}/scripts/pbt_sps_seed.sh"

GPU_ID="${1:-${GPU_ID:-0}}"
if [[ $# -ge 1 ]]; then
  shift
fi

MEM_MIB_MAX="${MEM_MIB_MAX:-512}"
UTIL_MAX="${UTIL_MAX:-5}"
IDLE_CHECKS="${IDLE_CHECKS:-2}"
POLL_SEC="${POLL_SEC:-30}"

if [[ ! -x "${SPS_SH}" ]] && [[ -f "${SPS_SH}" ]]; then
  chmod +x "${SPS_SH}"
fi
if [[ ! -f "${SPS_SH}" ]]; then
  echo "Missing ${SPS_SH}" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" >&2
  exit 1
fi

gpu_stats() {
  # prints: util_pct used_mib
  nvidia-smi -i "${GPU_ID}" --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1+0, $2+0}'
}

echo "Waiting for GPU ${GPU_ID} idle (mem<=${MEM_MIB_MAX} MiB, util<=${UTIL_MAX}%, ${IDLE_CHECKS}x) ..."
echo "Then: MODE=${MODE:-record} STRATEGY=${STRATEGY:-prioritized} ${SPS_SH} ${GPU_ID} $*"

idle_streak=0
while true; do
  read -r util mem < <(gpu_stats)
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  if (( util <= UTIL_MAX && mem <= MEM_MIB_MAX )); then
    idle_streak=$((idle_streak + 1))
    echo "[${ts}] GPU${GPU_ID}: util=${util}% mem=${mem}MiB  idle ${idle_streak}/${IDLE_CHECKS}"
    if (( idle_streak >= IDLE_CHECKS )); then
      break
    fi
  else
    if (( idle_streak > 0 )); then
      echo "[${ts}] GPU${GPU_ID}: util=${util}% mem=${mem}MiB  (reset idle streak)"
    else
      echo "[${ts}] GPU${GPU_ID}: util=${util}% mem=${mem}MiB  busy"
    fi
    idle_streak=0
  fi
  sleep "${POLL_SEC}"
done

echo "GPU ${GPU_ID} idle — starting pbt_sps_seed.sh"
exec env GPU_ID="${GPU_ID}" "${SPS_SH}" "${GPU_ID}" "$@"
