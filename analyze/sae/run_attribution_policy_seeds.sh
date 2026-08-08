#!/usr/bin/env bash
# Canonical policy-seed attribution pipeline.
#
# For each aligned seed triplet (ReCord, Reactive, Self-play):
#   1) collect shared-obs activations  (skip if present)
#   2) train independent SAE per method (skip if ckpt present)
#   3) retrieval → feature matching → semantics
#   4) projection attribution validation
# Then:
#   5) seed-level aggregate stats → seed_level_summary.json
#
# Usage:
#   ./analyze/sae/run_attribution_policy_seeds.sh [GPU_ID]
#   ./analyze/sae/run_attribution_policy_seeds.sh 1
#   FORCE=1 ./analyze/sae/run_attribution_policy_seeds.sh
#   SKIP_ATTR=1 ./analyze/sae/run_attribution_policy_seeds.sh   # collect+SAE only

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"

BASE_PATH="${BASE_PATH:-/data/puffer/experiments}"
PROBE_STEP="${PROBE_STEP:-1908}"
probe_pad=$(printf '%06d' "$PROBE_STEP")
GPU_ID="${1:-${GPU_ID:-0}}"
DEVICE="${DEVICE:-cuda}"
TOP_K="${TOP_K:-5}"
MAX_ROWS="${MAX_ROWS:-2048}"
FORCE="${FORCE:-0}"
FORCE_COLLECT="${FORCE_COLLECT:-0}"
FORCE_SAE="${FORCE_SAE:-0}"
FORCE_ANALYSIS="${FORCE_ANALYSIS:-0}"
SKIP_ATTR="${SKIP_ATTR:-0}"
CKPT_NAME="${CKPT_NAME:-sae_step_0001000.pt}"

TRAIN_NUM_MAPS="${TRAIN_NUM_MAPS:-10000}"
VAL_NUM_MAPS="${VAL_NUM_MAPS:-1500}"
MAX_MIN_DIST_M="${MAX_MIN_DIST_M:--1}"
MIN_OTHER_SPEED_MPS="${MIN_OTHER_SPEED_MPS:-0.5}"
MIN_EGO_SPEED_MPS="${MIN_EGO_SPEED_MPS:-0.0}"
MAX_TIMESTEPS_PER_PAIR="${MAX_TIMESTEPS_PER_PAIR:-4}"
MIN_TIMESTEP_GAP="${MIN_TIMESTEP_GAP:-8}"
MAX_SAMPLES_PER_SCENE="${MAX_SAMPLES_PER_SCENE:-64}"

ARCH="${ARCH:-topk}"
EXPANSION="${EXPANSION:-2}"
K="${K:-8}"
L1="${L1:-1e-3}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
NUM_STEPS="${NUM_STEPS:-1000}"
LR="${LR:-5e-4}"
SAVE_EVERY="${SAVE_EVERY:-1000}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-1000}"

OUT_ROOT="${OUT_ROOT:-/data/puffer/sae/runs/attribution_policy_seeds}"
EXPERIMENTS="selfplay,reactive_0.25,replay_0.25"

has_final_ckpt() {
  local exp="$1" id="$2"
  [[ -f "${BASE_PATH}/${exp}/puffer_drive_${id}.pt" ]]
}

list_seed_ckpts() {
  local exp="$1" f id
  for f in "$BASE_PATH/$exp"/puffer_drive_*.pt; do
    [[ -f "$f" ]] || continue
    id=$(basename "$f" .pt)
    id=${id#puffer_drive_}
    echo "$id"
  done | sort
}

policy_ckpt_path() {
  local exp="$1" id="$2"
  echo "${BASE_PATH}/${exp}/puffer_drive_${id}.pt"
}

act_path() {
  echo "$1/human_replay/$2/step_${probe_pad}/activations.npz"
}

activations_ready() {
  local root="$1"
  [[ -f "$(act_path "$root" training)" && -f "$(act_path "$root" validation)" ]]
}

sae_ready() {
  [[ -f "$1/$2/$CKPT_NAME" ]]
}

write_seed_info() {
  local path="$1"
  local idx="$2"
  local rec="$3" rea="$4" sp="$5"
  cat > "$path" <<EOF
{
  "seed_index": ${idx},
  "probe_step": ${PROBE_STEP},
  "policy_runs": {
    "record": "${rec}",
    "reactive": "${rea}",
    "selfplay": "${sp}"
  }
}
EOF
}

collect_mode() {
  local sae_root="$1" mode="$2" num_maps="$3" policy_dirs="$4" ref_run="$5"
  local force_flag=()
  if [[ "$FORCE_COLLECT" == "1" || "$FORCE" == "1" ]]; then
    force_flag+=(--force-collect)
  fi
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$SCRIPT_DIR/collect_sae_activations.py" \
    --base-path "$BASE_PATH" \
    --reference-exp replay_0.25 \
    --experiments "$EXPERIMENTS" \
    --run-dir "$ref_run" \
    --policy-run-dirs "$policy_dirs" \
    --output-dir "$sae_root" \
    --num-maps "$num_maps" \
    --device "$DEVICE" \
    --data-mode "$mode" \
    --probe-step "$PROBE_STEP" \
    --max-min-dist-m "$MAX_MIN_DIST_M" \
    --min-other-speed-mps "$MIN_OTHER_SPEED_MPS" \
    --min-ego-speed-mps "$MIN_EGO_SPEED_MPS" \
    --max-timesteps-per-pair "$MAX_TIMESTEPS_PER_PAIR" \
    --min-timestep-gap "$MIN_TIMESTEP_GAP" \
    --max-samples-per-scene "$MAX_SAMPLES_PER_SCENE" \
    --save-obs \
    "${force_flag[@]}"
}

train_seed_saes() {
  local sae_root="$1" sae_out="$2"
  mkdir -p "$sae_out"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$SCRIPT_DIR/train_sae.py" \
    --sae-root "$sae_root" \
    --probe-step "$PROBE_STEP" \
    --experiments "$EXPERIMENTS" \
    --architecture "$ARCH" \
    --expansion "$EXPANSION" \
    --k "$K" \
    --l1-coefficient "$L1" \
    --batch-size "$BATCH_SIZE" \
    --num-steps "$NUM_STEPS" \
    --lr "$LR" \
    --device "$DEVICE" \
    --save-every "$SAVE_EVERY" \
    --checkpoint-steps "$CHECKPOINT_STEPS" \
    --out-dir "$sae_out"
}

run_seed_matching() {
  local sae_root="$1" sae_run="$2" analysis_dir="$3"
  mkdir -p "$analysis_dir"
  echo "  retrieval"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$SCRIPT_DIR/top_activation_retrieval.py" \
    --run-dir "$sae_run" \
    --ckpt-name "$CKPT_NAME" \
    --sae-root "$sae_root" \
    --probe-step "$PROBE_STEP" \
    --data-mode validation \
    --out-dir "$analysis_dir" \
    --top-k 32 \
    --device "$DEVICE" \
    --skip-dead
  echo "  matching"
  "$PYTHON" "$SCRIPT_DIR/feature_matching.py" \
    --analysis-dir "$analysis_dir" \
    --out-dir "$analysis_dir/matching" \
    --normalize rank \
    --metric spearman \
    --match-method mutual_nn \
    --match-threshold 0.25
  echo "  semantics"
  "$PYTHON" "$SCRIPT_DIR/feature_semantics.py" \
    --analysis-dir "$analysis_dir" \
    --experiments record,reactive,selfplay \
    --out-dir "$analysis_dir/semantics"
}

# ---------- discover aligned seeds (flat puffer_drive_{id}.pt at experiment root) ----------
mapfile -t REC_IDS < <(list_seed_ckpts replay_0.25)
mapfile -t REA_IDS < <(list_seed_ckpts reactive_0.25)
mapfile -t SP_IDS < <(list_seed_ckpts selfplay)

n_rec=${#REC_IDS[@]}
n_rea=${#REA_IDS[@]}
n_sp=${#SP_IDS[@]}
n_seeds=$n_rec
(( n_rea < n_seeds )) && n_seeds=$n_rea
(( n_sp < n_seeds )) && n_seeds=$n_sp

mkdir -p "$OUT_ROOT"

echo "========== Policy-seed pipeline =========="
echo "  steps: collect → SAE → matching → attribution → seed stats"
echo "  probe_step=${PROBE_STEP}  n_seeds=${n_seeds}"
echo "  OUT_ROOT=${OUT_ROOT}"
for ((i = 0; i < n_seeds; i++)); do
  echo "  seed${i}: puffer_drive_${REC_IDS[$i]}.pt | puffer_drive_${REA_IDS[$i]}.pt | puffer_drive_${SP_IDS[$i]}.pt"
done
echo

if [[ "$n_seeds" -le 0 ]]; then
  echo "No aligned flat puffer_drive_*.pt checkpoints with probe metadata step ${probe_pad}."
  exit 1
fi

for ((i = 0; i < n_seeds; i++)); do
  rec_id="${REC_IDS[$i]}"
  rea_id="${REA_IDS[$i]}"
  sp_id="${SP_IDS[$i]}"
  rec="$(policy_ckpt_path replay_0.25 "$rec_id")"
  rea="$(policy_ckpt_path reactive_0.25 "$rea_id")"
  sp="$(policy_ckpt_path selfplay "$sp_id")"
  seed_tag="seed${i}__rec_${rec_id}__rea_${rea_id}__sp_${sp_id}"
  seed_out="$OUT_ROOT/$seed_tag"
  sae_root="$seed_out"
  sae_run="$seed_out/sae_run"
  analysis_dir="$seed_out/analysis"
  val_out="$seed_out/validation"
  policy_dirs="record=${rec};reactive=${rea};selfplay=${sp}"

  echo "############################################################"
  echo "# [${i}/$((n_seeds - 1))] ${seed_tag}"
  echo "############################################################"

  if [[ "$FORCE" != "1" && "$SKIP_ATTR" != "1" && -f "$val_out/attribution_validation_summary.json" ]]; then
    echo "SKIP (attribution already done)"
    continue
  fi

  mkdir -p "$seed_out"
  write_seed_info "$seed_out/seed_info.json" "$i" "$rec_id" "$rea_id" "$sp_id"

  # 1) collect
  if [[ "$FORCE" == "1" || "$FORCE_COLLECT" == "1" ]] || ! activations_ready "$sae_root"; then
    echo "[1/4] Collect activations"
    collect_mode "$sae_root" training "$TRAIN_NUM_MAPS" "$policy_dirs" "$rec"
    collect_mode "$sae_root" validation "$VAL_NUM_MAPS" "$policy_dirs" "$rec"
  else
    echo "[1/4] SKIP collect (activations present)"
  fi

  # 2) SAE
  need_sae=0
  for exp in selfplay reactive_0.25 replay_0.25; do
    if [[ "$FORCE" == "1" || "$FORCE_SAE" == "1" ]] || ! sae_ready "$sae_run" "$exp"; then
      need_sae=1
      break
    fi
  done
  if [[ "$need_sae" == "1" ]]; then
    echo "[2/4] Train SAE (3 methods, this seed)"
    train_seed_saes "$sae_root" "$sae_run"
  else
    echo "[2/4] SKIP SAE (${CKPT_NAME} present)"
  fi

  if [[ "$SKIP_ATTR" == "1" ]]; then
    echo "SKIP_ATTR=1 — stop after SAE"
    continue
  fi

  # 3) matching
  if [[ "$FORCE" == "1" || "$FORCE_ANALYSIS" == "1" || ! -f "$analysis_dir/matching/triplet_matches.json" ]]; then
    echo "[3/4] Retrieval + matching + semantics"
    run_seed_matching "$sae_root" "$sae_run" "$analysis_dir"
  else
    echo "[3/4] SKIP matching (triplet_matches present)"
  fi

  # 4) attribution
  echo "[4/4] Projection attribution"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$SCRIPT_DIR/projection_attribution_validation.py" \
    --analysis-dir "$analysis_dir" \
    --run-dir "$sae_run" \
    --ckpt-name "$CKPT_NAME" \
    --sae-root "$sae_root" \
    --probe-step "$PROBE_STEP" \
    --top-k "$TOP_K" \
    --max-rows "$MAX_ROWS" \
    --pool-modes "max,winning_slot" \
    --policy-run-dirs "$policy_dirs" \
    --out-dir "$val_out" \
    --skip-fd
  echo
done

# 5) seed-level stats (always refresh from whatever seeds finished)
echo "========== Seed-level aggregation =========="
"$PYTHON" "$SCRIPT_DIR/aggregate_seed_attribution.py" --out-root "$OUT_ROOT"
"$PYTHON" "$SCRIPT_DIR/correlate_attribution_logreplay.py" --out-root "$OUT_ROOT"

echo "Done."
echo "  per-seed: ${OUT_ROOT}/seed*/{human_replay,sae_run,analysis,validation}/"
echo "  summary:  ${OUT_ROOT}/seed_level_summary.json"
echo "  corr:     ${OUT_ROOT}/attribution_logreplay_correlation.json"
