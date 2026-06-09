#!/usr/bin/env bash
# Usage: ./tmp3.sh <cuda_device> <start_idx> <end_idx> [num_collect_rollout]
# Example: ./tmp3.sh 0 0 25 50
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <cuda_device> <collect_start_idx> <collect_end_idx> [num_collect_rollout]" >&2
  echo "  num_collect_rollout defaults to 50" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${1}"
START="${2}"
END="${3}"
NUM="${4:-50}"

export CUDA_VISIBLE_DEVICES
exec puffer zeroshot puffer_drive \
  --pbt.num-collect-rollout="${NUM}" \
  --pbt.collect-start-idx="${START}" \
  --pbt.collect-end-idx="${END}"
