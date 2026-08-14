#!/usr/bin/env bash
# Criticality-metrics report (Westhofen et al. 2022 survey) across record/reactive/selfplay,
# applied post-hoc to the packs already written by run_coordination.sh and run_ego_readout.sh.
#
# Usage:
#   ./analyze/coordination/run_criticality_report.sh
#   OUT_ROOT=/data/puffer/results/coordination ./analyze/coordination/run_criticality_report.sh
#
# Requires run_coordination.sh and/or run_ego_readout.sh to have already produced packs
# under OUT_ROOT. Pure CPU post-hoc analysis -- no GPU, no policy loading.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"
cd "$REPO_ROOT"

OUT_ROOT="${OUT_ROOT:-/data/puffer/results/coordination}"

echo "========== Criticality metrics report =========="
echo "  out=${OUT_ROOT}/criticality_report/"

"$PYTHON" "$SCRIPT_DIR/criticality_report.py" --out-root "$OUT_ROOT"