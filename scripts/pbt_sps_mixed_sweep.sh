#!/usr/bin/env bash
# Sweep hybrid (mixed) SPS over partner_replay_prob, plus pure endpoints.
#
# Runs scripts/pbt_sps_seed.sh for:
#   MODE=reactive                         (p = 0)
#   MODE=mixed PARTNER_REPLAY_PROB=...    (interior points)
#   MODE=record                           (p = 1)
#
# Usage:
#   STRATEGY=uniform ./scripts/pbt_sps_mixed_sweep.sh 0
#   STRATEGY=uniform PROBS="0.25 0.5 0.75" SEEDS="42 3" ./scripts/pbt_sps_mixed_sweep.sh 0
#   STRATEGY=uniform SKIP_ENDPOINTS=1 PROBS="0.5" ./scripts/pbt_sps_mixed_sweep.sh 1
#
# Env:
#   STRATEGY          uniform | prioritized | curriculum  (default: uniform)
#   PROBS             space-separated interior probs (default: "0.25 0.5 0.75")
#   SKIP_ENDPOINTS    if 1, skip pure reactive/record (default: 0)
#   SEEDS / TOTAL_TIMESTEPS / POP_PATH — forwarded to pbt_sps_seed.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPS_SH="${ROOT}/scripts/pbt_sps_seed.sh"
GPU_ID="${1:-0}"
STRATEGY="${STRATEGY:-uniform}"
PROBS="${PROBS:-0.25 0.5 0.75}"
SKIP_ENDPOINTS="${SKIP_ENDPOINTS:-0}"

if [[ ! -f "${SPS_SH}" ]]; then
  echo "Missing ${SPS_SH}" >&2
  exit 1
fi
chmod +x "${SPS_SH}" 2>/dev/null || true

echo "========== Hybrid SPS sweep =========="
echo "  GPU=${GPU_ID}  STRATEGY=${STRATEGY}"
echo "  PROBS=${PROBS}"
echo "  SKIP_ENDPOINTS=${SKIP_ENDPOINTS}"
echo "======================================"

run_one() {
  echo ""
  echo ">>> $*"
  env "$@" "${SPS_SH}" "${GPU_ID}"
}

if [[ "${SKIP_ENDPOINTS}" != "1" ]]; then
  run_one MODE=reactive STRATEGY="${STRATEGY}"
fi

read -r -a PROB_LIST <<< "${PROBS}"
for p in "${PROB_LIST[@]}"; do
  run_one MODE=mixed PARTNER_REPLAY_PROB="${p}" STRATEGY="${STRATEGY}"
done

if [[ "${SKIP_ENDPOINTS}" != "1" ]]; then
  run_one MODE=record STRATEGY="${STRATEGY}"
fi

echo ""
echo "Hybrid SPS sweep done. Summaries under /data/puffer/experiments/sps_${STRATEGY}_*/"
echo "Plot:"
echo "  python analyze/plot_sps.py --mixed --strategy ${STRATEGY}"
