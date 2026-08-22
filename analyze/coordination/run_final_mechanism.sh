#!/usr/bin/env bash
# ============================================================
# ReCord vs Reactive — Final Mechanistic Analysis
#
# Final claim:
#   ReCord's recurrent dynamics repeatedly generate a transient
#   interaction-sensitive component during coordination-critical
#   states. This component causally biases immediate action
#   selection toward more conservative responses, but is not
#   itself a persistent plan.
#
# Shorter:
#   ReCord induces a reproducible interaction-sensitive recurrent
#   computation that is selectively recruited during critical
#   interactions and causally shifts immediate control toward
#   more conservative actions.
#
# Canonical entry point (read this file to understand the pipeline).
# Usage:
#   bash analyze/coordination/run_final_mechanism.sh help
#   bash analyze/coordination/run_final_mechanism.sh check_results
#   bash analyze/coordination/run_final_mechanism.sh smoke
#   FORCE=1 bash analyze/coordination/run_final_mechanism.sh <stage>
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1

# ============================================================
# 0. Frozen configuration
# ============================================================
TRAIN_BINARIES="${TRAIN_BINARIES:-/data/puffer/resources/drive/binaries/training}"
VAL_BINARIES="${VAL_BINARIES:-/data/puffer/resources/drive/binaries/validation}"

RESULT_10K="${RESULT_10K:-/data/puffer/crosscoder/crosscoder_10k}"
RESULT_ROOT="${RESULT_ROOT:-/data/puffer/crosscoder/crosscoder_mechanism}"
# Historical immutable trees (read-only reference):
#   /data/puffer/results/crosscoder{,_10k,_mechanism}/

DICT_SIZE=1024
LAMBDA=0.3
SUBSPACE_K=32
ALPHA=0.5
ALPHAS="0.25,0.5,0.75,1.0"
HIDDEN=256

NUM_TRAIN_MAPS="${NUM_TRAIN_MAPS:-10000}"
NUM_VAL_MAPS="${NUM_VAL_MAPS:-10000}"
DEV_MAPS="${DEV_MAPS:-1000}"
SHARD_SIZE="${SHARD_SIZE:-1000}"
# maintain = primary (frozen gates). record|reactive = robustness. all = three dirs.
EGO_MODE="${EGO_MODE:-maintain}"
DRIVER_SEED="${DRIVER_SEED:-42}"

GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
FORCE="${FORCE:-0}"
PERSIST_MAPS="${PERSIST_MAPS:-1000}"

# Exact four matched train.seed pairs (from policy_pairs.json)
# seed:record_wandb:reactive_wandb
POLICY_PAIRS=(
  "42:yb0uds6n:xc4bxfyr"
  "3:i2oeu54g:t6u7nnbx"
  "11:w1wu2uom:5vf9d37q"
  "0:x9etv2km:757932iw"
)

EXPERIMENTS_ROOT="${EXPERIMENTS_ROOT:-/data/puffer/experiments}"
REC_DIR="$EXPERIMENTS_ROOT/record-uniform-popul_lane_nominal-wandb"
REA_DIR="$EXPERIMENTS_ROOT/reactive-uniform-popul_lane_nominal-wandb"

have() { [[ -e "$1" ]]; }
skip_msg() { echo "[reuse] $1 (set FORCE=1 to regenerate)"; }
run_py() {
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$@"
}

CC="$SCRIPT_DIR/crosscoder"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export FORCE

# ============================================================
# help
# ============================================================
help() {
  cat <<'EOF'
ReCord vs Reactive — final mechanistic pipeline (Gates 1, 2, 5A, 6 only)

  collect     ONE place: 10K train+val, 8 policies; EGO_MODE=maintain|record|reactive|all
  crosscoder  Optional: confirmatory Crosscoder train (reads collect; no re-collect)
  replicate   matched-seed |Δc| + subspace U (EGO_MODE=maintain|record|reactive|all)
  subspace    Gate 2
  intervene   Gate 5A
  transient   Gate 6 (H1–H5 + one-shot persistence)

  smoke | check_results | help

  bash analyze/coordination/run_final_mechanism.sh all
  bash analyze/coordination/run_final_mechanism.sh check_results
  EGO_MODE=all bash analyze/coordination/run_final_mechanism.sh collect

Active gates: 1 PASS | 2 PASS | 5A PASS (3/4) | 6 STRONG
Removed from pipeline: Gate 3/4 semantics, Gate 5B closed-loop
NOT supported: belief, intent, planning, world model, persistent latent plan
EOF
}

# ============================================================
# 1. Collect paired same-state activations
# ============================================================
collect() {
  # Input:  TRAIN/VAL binaries + 4 matched policy ckpts (8 policies)
  # Output (maintain / primary):
  #   $RESULT_ROOT/policy_seed_replication/{train,validation}_dataset/
  # Output (record|reactive robustness):
  #   $RESULT_ROOT/policy_seed_replication/ego_{record,reactive}/...
  # All modes: identical obs history within a run; both policies still forward.
  # EGO_MODE=maintain|record|reactive|all  DRIVER_SEED=42 (who drives for biased modes)
  echo "========== collect ego_mode=$EGO_MODE =========="
  modes=()
  if [[ "$EGO_MODE" == "all" ]]; then
    modes=(maintain record reactive)
  else
    modes=("$EGO_MODE")
  fi
  need_run=0
  for m in "${modes[@]}"; do
    if [[ "$m" == "maintain" ]]; then
      val_shards="$RESULT_ROOT/policy_seed_replication/validation_dataset/shards"
    else
      val_shards="$RESULT_ROOT/policy_seed_replication/ego_${m}/validation_dataset/shards"
    fi
    if [[ "$FORCE" == "1" ]] || ! have "$val_shards"; then
      need_run=1
      break
    fi
    skip_msg "$m activations already present"
  done
  if [[ "$need_run" != "1" ]]; then
    return 0
  fi
  force_flag=()
  [[ "$FORCE" == "1" ]] && force_flag+=(--force)
  run_py "$CC/collect.py" \
    --out-root "$RESULT_ROOT" \
    --num-train-maps "$NUM_TRAIN_MAPS" \
    --num-val-maps "$NUM_VAL_MAPS" \
    --shard-size "$SHARD_SIZE" \
    --device "$DEVICE" \
    --ego-mode "$EGO_MODE" \
    --driver-seed "$DRIVER_SEED" \
    "${force_flag[@]}"
}

# ============================================================
# 2. Train/refit frozen sparse Crosscoder (confirmatory 10K/10K)
# ============================================================
crosscoder() {
  # Input:  activations from collect for EGO_MODE (default maintain)
  # Output: maintain → $RESULT_10K ; record/reactive → ${RESULT_10K}_ego_{mode}
  # Q: Is sparse |Δc| tight>nominal (~1.5×) while normalized raw hidden ≈1.0?
  if [[ "$EGO_MODE" == "all" ]]; then
    echo "crosscoder: EGO_MODE=all not supported; run once per mode" >&2
    return 1
  fi
  local out="$RESULT_10K"
  if [[ "$EGO_MODE" != "maintain" ]]; then
    out="${RESULT_10K}_ego_${EGO_MODE}"
  fi
  echo "========== crosscoder (10K/10K) ego_mode=$EGO_MODE =========="
  echo "  frozen: dict=$DICT_SIZE  lambda=$LAMBDA  independent ISTA"
  echo "  out=$out"
  if [[ "$FORCE" != "1" ]] && have "$out/summary.json"; then
    skip_msg "$out/summary.json"
    return 0
  fi
  if [[ "$EGO_MODE" == "maintain" ]]; then
    need="$RESULT_ROOT/policy_seed_replication/train_dataset/shards"
  else
    need="$RESULT_ROOT/policy_seed_replication/ego_${EGO_MODE}/train_dataset/shards"
  fi
  if ! have "$need"; then
    echo "Need collect for ego_mode=$EGO_MODE first ($need)." >&2
    collect
  fi
  run_py "$CC/pipeline.py" \
    --out-root "$out" \
    --acts-root "$RESULT_ROOT/policy_seed_replication" \
    --ego-mode "$EGO_MODE" \
    --rec-key record_s3 \
    --rea-key reactive_s11 \
    --epochs 25 \
    --seeds 1 \
    --device "$DEVICE"
}

# ============================================================
# 3–4. Replicate interaction-specific latent divergence
# ============================================================
replicate() {
  # pairs → replicate → subspace → report (EGO_MODE selects collect tree)
  # maintain → $RESULT_ROOT; record/reactive → ${RESULT_ROOT}_ego_{mode}
  echo "========== replicate + subspace ego_mode=$EGO_MODE =========="
  if [[ "$EGO_MODE" == "all" ]]; then
    bash "$CC/run_replicate_all_ego.sh"
    return 0
  fi
  if [[ "$EGO_MODE" == "maintain" ]]; then
    out="$RESULT_ROOT"
    need="$RESULT_ROOT/policy_seed_replication/train_dataset/shards"
  else
    out="${RESULT_ROOT}_ego_${EGO_MODE}"
    need="$RESULT_ROOT/policy_seed_replication/ego_${EGO_MODE}/train_dataset/shards"
  fi
  if [[ "$FORCE" != "1" ]] && have "$out/summary.json" && have "$out/subspace/summary.json"; then
    skip_msg "replicate+subspace already present ($out)"
    return 0
  fi
  if ! have "$need"; then
    echo "Need collect for ego_mode=$EGO_MODE first ($need)." >&2
    collect
  fi
  run_py "$CC/replicate.py" \
    --out-root "$RESULT_ROOT" \
    --acts-root "$RESULT_ROOT/policy_seed_replication" \
    --ego-mode "$EGO_MODE" \
    --device "$DEVICE" \
    --epochs 20 \
    --cc-seeds 5
}

# ============================================================
# 5. Interaction-sensitive consensus subspace (included in replicate)
# ============================================================
subspace() {
  echo "========== subspace (via replicate) =========="
  if [[ "$FORCE" != "1" ]] && have "$RESULT_ROOT/subspace/consensus/matched_seed_0_U.npy"; then
    skip_msg "subspace/consensus/*.npy"
    return 0
  fi
  replicate
}

# ============================================================
# 6. Immediate same-state causal intervention (Gate 5A)
# ============================================================
intervene() {
  # Projector: delta_U = U U^T (h_R - h_Rea);  h' = h_Rea + α delta_U
  # Output: $RESULT_ROOT/intervention/summary.json
  # Q: Does the patch move Reactive actions toward ReCord (ΔKL>0), esp. tight?
  echo "========== intervene (Gate 5A, primary α=$ALPHA) =========="
  echo "  alpha sweep: {$ALPHAS}"
  if [[ "$FORCE" != "1" ]] && have "$RESULT_ROOT/intervention/summary.json"; then
    skip_msg "intervention/summary.json"
    return 0
  fi
  run_py "$CC/intervention.py" --out-root "$RESULT_ROOT" --device "$DEVICE"
}

# ============================================================
# 7. Transient causal characterization H1–H5 (Gate 6)
# ============================================================
transient() {
  # H1 sparse ΔKL | H2 criticality/event | H3 natural regen vs 1-step overwrite
  # H4 more braking / less accel | H5 interaction > controls
  echo "========== transient (Gate 6 H1–H5) =========="
  if [[ "$FORCE" == "1" ]] || ! have "$RESULT_ROOT/intervention_persistence/summary.json"; then
    echo "  [one-shot persistence — artificial half-life ≈ 1]"
    run_py "$CC/persistence.py" --out-root "$RESULT_ROOT" --maps "$PERSIST_MAPS" --device "$DEVICE"
  else
    skip_msg "intervention_persistence/summary.json"
  fi
  if [[ "$FORCE" != "1" ]] && have "$RESULT_ROOT/transient_causal_characterization/summary.json"; then
    skip_msg "transient_causal_characterization/summary.json"
    return 0
  fi
  run_py "$CC/transient_causal.py" --out-root "$RESULT_ROOT" --device "$DEVICE"
}

# (Gate 5B closed-loop removed from active pipeline)


# ============================================================
# smoke — lightweight regression (no full 10K)
# ============================================================
smoke() {
  echo "========== smoke =========="
  export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
  run_py "$SCRIPT_DIR/crosscoder/test_cleanup_regression.py"
  run_py - <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path("analyze/coordination").resolve()))
from pathlib import Path
import json
import numpy as np
from crosscoder.intervention import project, load_pair_bases
from crosscoder.frozen_config import FROZEN_K, HIDDEN_DIM, PRIMARY_ALPHA

root = Path("/data/puffer/crosscoder/crosscoder_mechanism")
pairs = json.loads((root / "policy_seed_replication" / "policy_pairs.json").read_text())
pid = pairs["selected_pairs"][0]["pair_id"]
b = load_pair_bases(root, pid)
assert b["U"].shape[0] == HIDDEN_DIM
rng = np.random.default_rng(0)
delta = rng.normal(size=(8, HIDDEN_DIM))
d1 = project(delta, b["U"])
assert np.allclose(project(d1, b["U"]), d1, atol=1e-5)
print(f"PASS bases {pid} U={b['U'].shape} controls={b['controls_dir']} alpha={PRIMARY_ALPHA}")
pers = json.loads((root / "intervention_persistence" / "summary.json").read_text())
hls = [pers["one_shot"][p]["half_life_RU"] for p in pers["one_shot"]]
assert all(h == 1 or h <= 2 for h in hls), hls
print(f"PASS artificial half-lives={hls}")
g6 = json.loads((root / "transient_causal_characterization" / "summary.json").read_text())
across = g6.get("across_pairs") or {}
assert "cum10_positive_mass" in across
print(f"PASS transient across_pairs keys={list(across)}")
print("smoke OK")
PY
}

# ============================================================
# check_results — read frozen summaries only
# ============================================================
check_results() {
  echo "========== check_results (no rerun) =========="
  RESULT_ROOT="$RESULT_ROOT" RESULT_10K="$RESULT_10K" run_py - <<'PY'
import json
from pathlib import Path
root = Path(__import__("os").environ["RESULT_ROOT"])
r10 = Path(__import__("os").environ["RESULT_10K"])

def load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None

top = load(root / "summary.json") or {}
rep = load(root / "policy_seed_replication" / "summary.json") or {}
sub = load(root / "subspace" / "summary.json") or load(root / "subspace" / "stability.json") or {}
inter = load(root / "intervention" / "summary.json") or {}
g6 = load(root / "transient_causal_characterization" / "summary.json") or {}
pers = load(root / "intervention_persistence" / "summary.json") or {}

print(f"\n[paths] mechanism={root}  10k={r10}")

print("\n[replication]")
across = (top.get("policy_seed_replication") or rep.get("across_pairs") or {})
dc = across.get("d_c_tight_over_nominal") or {}
print(f"  |Δc| tight/nom mean={dc.get('mean')} per_pair={dc.get('per_pair')}")
sf = across.get("scene_frac_positive_d_c") or {}
print(f"  scene+ mean={sf.get('mean')} per_pair={sf.get('per_pair')}")

print("\n[subspace]")
print(f"  mean_overlap={sub.get('mean_overlap')} frozen_k={sub.get('frozen_k')}")
print(f"  per_pair_overlap={sub.get('per_pair_overlap')}")

print("\n[intervention]")
ver = inter.get("verdict") or (inter.get("gates") or {}).get("gate_5a_details") or {}
print(f"  status={ver.get('status') if isinstance(ver, dict) else ver}")
print(f"  n_passing={ver.get('n_passing') if isinstance(ver, dict) else None}")

print("\n[transient]")
across6 = (g6.get("across_pairs") or {})
print(f"  cum10={across6.get('cum10_positive_mass')}")
print(f"  nat_dur={across6.get('natural_elevated_duration')}")
if pers:
    hls = [pers["one_shot"][p]["half_life_RU"] for p in pers.get("one_shot", {})]
    print(f"  artificial R_U half-life: {hls}")
print()
PY
}

all_stages() {
  collect
  replicate
  subspace
  intervene
  transient
}

# ============================================================
# dispatch
# ============================================================
cmd="${1:-help}"
case "$cmd" in
  help|-h|--help) help ;;
  all) all_stages ;;
  collect) collect ;;
  crosscoder) crosscoder ;;
  replicate) replicate ;;
  subspace) subspace ;;
  intervene) intervene ;;
  transient) transient ;;
  smoke) smoke ;;
  check_results) check_results ;;
  *)
    echo "Unknown stage: $cmd" >&2
    help
    exit 1
    ;;
esac
