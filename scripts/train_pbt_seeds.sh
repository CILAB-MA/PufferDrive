#!/usr/bin/env bash
# Unified Drive-PBT multi-seed train: MODE × STRATEGY × population.
#
# Usage:
#   ./scripts/train_pbt_seeds.sh [GPU_ID]
#   MODE=record STRATEGY=prioritized ./scripts/train_pbt_seeds.sh 0
#   MODE=reactive STRATEGY=uniform POP_PATH=/data/puffer/popul_lane_nominal ./scripts/train_pbt_seeds.sh 1
#   MODE=record STRATEGY=curriculum SEEDS="42" CURRICULUM_STEPS=100 ./scripts/train_pbt_seeds.sh 0
#
# Checkpoints land in /data/puffer/experiments/{replay|reactive}_{strategy}_{pop_short}/
# so analyze/zero_shot.sh can pick them up as FOLDER, e.g.:
#   TRAIN_MODE=reactive STRATEGY=uniform ./analyze/zero_shot.sh 0
#   ./analyze/zero_shot.sh 0 replay_prioritized_nominal popul_lane_nominal
#
# Env:
#   MODE              record | reactive          (default: record)
#   STRATEGY          uniform | prioritized | curriculum  (default: prioritized)
#   POP_PATH          population corpus dir
#                     default: curriculum → /data/puffer/popul_curriculum
#                              else       → /data/puffer/popul_lane_nominal
#   TYPES_PATH        difficulty types (.npy/.json); default $POP_PATH/saved/difficulty_types.npy
#   CURRICULUM_STEPS  default 10000 (curriculum only)
#   NUM_COMBINATION   default 10
#   SCORE_TRANSFORM   default rank_low (prioritized only)
#   DATA_DIR          default /data/puffer/experiments/${EXP_NAME}/
#   EXP_NAME          default {replay|reactive}_{strategy}_{pop_short}
#                     pop_short: popul_lane_nominal → nominal, popul_mix → mix
#                     (same FOLDER as analyze/zero_shot.sh)
#   SEEDS             default "42 3 11"
#   WANDB_ENTITY      default cilab-ma
#   WANDB_PROJECT     default puffer-drive-icra
set -euo pipefail

GPU_ID="${1:-0}"
SEEDS_ARG="${2:-}"
MODE="${MODE:-record}"
STRATEGY="${STRATEGY:-prioritized}"
NUM_COMBINATION="${NUM_COMBINATION:-10}"
CURRICULUM_STEPS="${CURRICULUM_STEPS:-10000}"
SCORE_TRANSFORM="${SCORE_TRANSFORM:-rank_low}"
WANDB_ENTITY="${WANDB_ENTITY:-cilab-ma}"
WANDB_PROJECT="${WANDB_PROJECT:-puffer-drive-icra}"
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
  plr)
    # alias kept for old script muscle memory
    STRATEGY="prioritized"
    ;;
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
# popul_lane_nominal → nominal, popul_mix → mix, popul_curriculum → curriculum
pop_body="${POP_NAME#popul_}"
POP_SHORT="${pop_body##*_}"
TYPES_PATH="${TYPES_PATH:-${POP_PATH}/saved/difficulty_types.npy}"

# Match analyze/zero_shot.sh FOLDER: replay_uniform_nominal, reactive_prioritized_mix, ...
EXP_NAME="${EXP_NAME:-${PBT_MODE}_${STRATEGY}_${POP_SHORT}}"
DATA_DIR="${DATA_DIR:-/data/puffer/experiments/${EXP_NAME}/}"

# --- preflight ---
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
    echo "  (collect + merge, e.g. analyze/data_concat.py)" >&2
    exit 1
  fi
else
  if [[ ! -f "${POP_PATH}/saved/population_keys.npy" ]]; then
    echo "Reactive mode requires ${POP_PATH}/saved/population_keys.npy" >&2
    exit 1
  fi
  # Prefer manifest-resolved source .pt dirs; fall back to local *.pt
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
    echo "Collect then merge first:" >&2
    echo "  NUM_CKPTS=4 ./analyze/collect_curriculum_rollout.sh ${GPU_ID}" >&2
    echo "  python analyze/data_concat.py --population-path ${POP_PATH} --total-rollouts 50" >&2
    exit 1
  fi
fi

echo "========== Drive-PBT train =========="
echo "  exp=${EXP_NAME}"
echo "  mode=${MODE} (pbt_mode=${PBT_MODE})"
echo "  strategy=${STRATEGY}"
echo "  population=${POP_PATH}"
echo "  data_dir=${DATA_DIR}"
echo "  num_combination=${NUM_COMBINATION}"
if [[ "${STRATEGY}" == "curriculum" ]]; then
  echo "  types=${TYPES_PATH}"
  echo "  curriculum_steps=${CURRICULUM_STEPS}"
elif [[ "${STRATEGY}" == "prioritized" ]]; then
  echo "  score_transform=${SCORE_TRANSFORM}"
fi
echo "  seeds=${SEED_LIST[*]}"
echo "  wandb=${WANDB_ENTITY}/${WANDB_PROJECT}  group=${EXP_NAME}"
echo "===================================="

for SEED in "${SEED_LIST[@]}"; do
  echo ""
  echo ">>> seed=${SEED}  group=${EXP_NAME}"

  CMD=(
    puffer train_pbt puffer_drive_pbt
    --pbt.pbt-mode "${PBT_MODE}"
    --pbt.population-path "${POP_PATH}"
    --pbt.strategy "${STRATEGY}"
    --pbt.num-combination "${NUM_COMBINATION}"
    --train.data-dir "${DATA_DIR}"
    --train.seed "${SEED}"
    --vec.seed "${SEED}"
    --eval.human-replay-eval True
    --wandb
    --wandb-entity "${WANDB_ENTITY}"
    --wandb-group "${EXP_NAME}"
    --wandb-project "${WANDB_PROJECT}"
  )

  if [[ "${STRATEGY}" == "prioritized" ]]; then
    CMD+=(--pbt.score-transform "${SCORE_TRANSFORM}")
  fi
  if [[ "${STRATEGY}" == "curriculum" ]]; then
    CMD+=(--pbt.curriculum-types-path "${TYPES_PATH}")
    CMD+=(--pbt.curriculum-steps "${CURRICULUM_STEPS}")
  fi

  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
done

echo ""
echo "All seeds finished. Checkpoints under: ${DATA_DIR}"
echo "Experiment: ${EXP_NAME}"
echo "Zero-shot:"
echo "  TRAIN_MODE=${MODE} EVAL_MODE=both STRATEGY=${STRATEGY} ./analyze/zero_shot.sh ${GPU_ID}"
echo "  ./analyze/zero_shot.sh ${GPU_ID} ${EXP_NAME} ${POP_NAME}"
