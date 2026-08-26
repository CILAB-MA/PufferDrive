#!/usr/bin/env bash
# Short Drive-PBT run for local SPS measurement (default 30M ego steps, no wandb).
# Same knobs as scripts/train_pbt_seeds.sh; map packing follows --train.seed / --vec.seed.
#
# Usage:
#   MODE=record STRATEGY=uniform ./scripts/pbt_sps_seed.sh 0
#   MODE=reactive STRATEGY=uniform ./scripts/pbt_sps_seed.sh 1
#
# Outputs under:
#   /data/puffer/experiments/sps_{STRATEGY}_{MODE}/
#     sps_seed_{SEED}.jsonl   — per-log SPS rows
#     sps_summary.json        — per-seed + overall mean SPS
#
# Defaults:
#   TOTAL_TIMESTEPS=30000000
#   SEEDS="42 3 11 0"
#   human-replay eval off
set -euo pipefail

GPU_ID="${1:-0}"
SEEDS_ARG="${2:-}"
MODE="${MODE:-record}"
STRATEGY="${STRATEGY:-prioritized}"
NUM_COMBINATION="${NUM_COMBINATION:-10}"
CURRICULUM_STEPS="${CURRICULUM_STEPS:-10000}"
SCORE_TRANSFORM="${SCORE_TRANSFORM:-rank_low}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000000}"
read -r -a SEED_LIST <<< "${SEEDS_ARG:-${SEEDS:-42 3 11 0}}"

if command -v python >/dev/null 2>&1; then
  PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
else
  echo "python/python3 not found on PATH" >&2
  exit 1
fi

case "${MODE}" in
  record) PBT_MODE="replay" ;;
  reactive) PBT_MODE="reactive" ;;
  *)
    echo "MODE must be record or reactive (got: ${MODE})" >&2
    exit 1
    ;;
esac

case "${STRATEGY}" in
  uniform|prioritized|curriculum) ;;
  plr) STRATEGY="prioritized" ;;
  *)
    echo "STRATEGY must be uniform, prioritized, or curriculum (got: ${STRATEGY})" >&2
    exit 1
    ;;
esac

if [[ "${STRATEGY}" == "curriculum" ]]; then
  POP_PATH="${POP_PATH:-${POPULATION_PATH:-/data/puffer/popul_curriculum}}"
else
  POP_PATH="${POP_PATH:-${POPULATION_PATH:-/data/puffer/popul_lane_nominal}}"
fi
POP_PATH="${POP_PATH%/}"
POP_NAME="$(basename "${POP_PATH}")"
TYPES_PATH="${TYPES_PATH:-${POP_PATH}/saved/difficulty_types.npy}"

# e.g. /data/puffer/experiments/sps_uniform_record
EXP_NAME="${EXP_NAME:-sps_${STRATEGY}_${MODE}}"
DATA_DIR="${DATA_DIR:-/data/puffer/experiments/${EXP_NAME}/}"
mkdir -p "${DATA_DIR}"

if [[ ! -d "${POP_PATH}" ]]; then
  echo "Population directory not found: ${POP_PATH}" >&2
  exit 1
fi
if [[ ! -f "${POP_PATH}/saved/global_ids.npy" ]]; then
  echo "Missing ${POP_PATH}/saved/global_ids.npy" >&2
  exit 1
fi

if [[ "${MODE}" == "record" ]]; then
  if [[ ! -f "${POP_PATH}/saved/other_actions_actions.npy" ]]; then
    echo "Record mode requires ${POP_PATH}/saved/other_actions_actions.npy" >&2
    exit 1
  fi
else
  if [[ ! -f "${POP_PATH}/saved/population_keys.npy" ]]; then
    echo "Reactive mode requires ${POP_PATH}/saved/population_keys.npy" >&2
    exit 1
  fi
  MANIFEST="${POP_PATH}/saved/population_manifest.json"
  if [[ ! -f "${MANIFEST}" ]]; then
    MANIFEST="${POP_PATH}/population_manifest.json"
  fi
  if [[ -f "${MANIFEST}" ]]; then
    if ! "${PYTHON}" - "${MANIFEST}" "${POP_PATH}" <<'PY'
import json, os, sys
man_path, pop = sys.argv[1], sys.argv[2]
with open(man_path) as f:
    keys = json.load(f).get("keys") or []
if not keys:
    raise SystemExit(1)
for e in keys:
    ckpt = e["checkpoint"]
    src = os.path.join(e.get("population") or pop, ckpt)
    alt = os.path.join(pop, ckpt)
    if not (os.path.isfile(src) or os.path.isfile(alt)):
        print(f"Missing reactive checkpoint: {src}", file=sys.stderr)
        raise SystemExit(1)
PY
    then
      echo "Reactive manifest checkpoints missing (see above)" >&2
      exit 1
    fi
  elif ! compgen -G "${POP_PATH}/*.pt" > /dev/null; then
    echo "Reactive mode requires *.pt in ${POP_PATH} or a population_manifest.json" >&2
    exit 1
  fi
fi

if [[ "${STRATEGY}" == "curriculum" ]]; then
  if [[ ! -f "${TYPES_PATH}" ]]; then
    echo "Missing ${TYPES_PATH}" >&2
    exit 1
  fi
fi

echo "========== Drive-PBT SPS bench =========="
echo "  exp=${EXP_NAME}"
echo "  mode=${MODE} (pbt_mode=${PBT_MODE})"
echo "  strategy=${STRATEGY}"
echo "  population=${POP_PATH}"
echo "  data_dir=${DATA_DIR}"
echo "  num_combination=${NUM_COMBINATION}"
echo "  total_timesteps=${TOTAL_TIMESTEPS}"
if [[ "${STRATEGY}" == "curriculum" ]]; then
  echo "  types=${TYPES_PATH}"
  echo "  curriculum_steps=${CURRICULUM_STEPS}"
elif [[ "${STRATEGY}" == "prioritized" ]]; then
  echo "  score_transform=${SCORE_TRANSFORM}"
fi
echo "  seeds=${SEED_LIST[*]}"
echo "  wandb=off  (local sps_seed_*.jsonl)"
echo "========================================="

for SEED in "${SEED_LIST[@]}"; do
  echo ""
  echo ">>> seed=${SEED}  dir=${DATA_DIR}  steps=${TOTAL_TIMESTEPS}"
  SPS_LOG="${DATA_DIR}/sps_seed_${SEED}.jsonl"
  : > "${SPS_LOG}"

  CMD=(
    puffer train_pbt puffer_drive_pbt
    --pbt.pbt-mode "${PBT_MODE}"
    --pbt.population-path "${POP_PATH}"
    --pbt.strategy "${STRATEGY}"
    --pbt.num-combination "${NUM_COMBINATION}"
    --train.data-dir "${DATA_DIR}"
    --train.seed "${SEED}"
    --vec.seed "${SEED}"
    --train.total-timesteps "${TOTAL_TIMESTEPS}"
    --eval.human-replay-eval False
  )

  if [[ "${STRATEGY}" == "prioritized" ]]; then
    CMD+=(--pbt.score-transform "${SCORE_TRANSFORM}")
  fi
  if [[ "${STRATEGY}" == "curriculum" ]]; then
    CMD+=(--pbt.curriculum-types-path "${TYPES_PATH}")
    CMD+=(--pbt.curriculum-steps "${CURRICULUM_STEPS}")
  fi

  CUDA_VISIBLE_DEVICES="${GPU_ID}" PUFFER_SPS_LOG="${SPS_LOG}" "${CMD[@]}"
done

# Aggregate mean SPS per seed (drop early warmup: first 20% of rows) + overall.
"${PYTHON}" - "${DATA_DIR}" "${MODE}" "${STRATEGY}" "${TOTAL_TIMESTEPS}" <<'PY'
import json, glob, os, sys
data_dir, mode, strategy, total_ts = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
seed_stats = []
for path in sorted(glob.glob(os.path.join(data_dir, "sps_seed_*.jsonl"))):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        continue
    seed = rows[0].get("seed")
    # Use second half of the run for a stable SPS estimate.
    start = max(0, len(rows) // 2)
    sps_vals = [float(r["SPS"]) for r in rows[start:] if float(r.get("SPS") or 0) > 0]
    mean_sps = sum(sps_vals) / len(sps_vals) if sps_vals else 0.0
    last = rows[-1]
    seed_stats.append({
        "seed": seed,
        "n_logs": len(rows),
        "mean_sps_second_half": mean_sps,
        "last_sps": float(last.get("SPS") or 0),
        "agent_steps": int(last.get("agent_steps") or 0),
        "uptime": float(last.get("uptime") or 0),
        "log": os.path.basename(path),
    })
overall = 0.0
if seed_stats:
    overall = sum(s["mean_sps_second_half"] for s in seed_stats) / len(seed_stats)
summary = {
    "mode": mode,
    "strategy": strategy,
    "total_timesteps": total_ts,
    "data_dir": data_dir,
    "seeds": seed_stats,
    "mean_sps_across_seeds": overall,
}
out = os.path.join(data_dir, "sps_summary.json")
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
print(f"Wrote {out}")
PY

echo ""
echo "SPS bench finished under: ${DATA_DIR}"
echo "  per-seed: ${DATA_DIR}sps_seed_*.jsonl"
echo "  summary:  ${DATA_DIR}sps_summary.json"
