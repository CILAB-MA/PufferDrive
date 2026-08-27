#!/usr/bin/env bash
# ============================================================
# ReCord vs Reactive — Final Mechanistic Analysis
#
# Canonical entry:
#   bash analyze/coordination/run_final_mechanism.sh all
#     → collect(maintain+record+reactive)
#     → crosscoder(all three)
#     → replicate+subspace(all three)
#     → intervene + persistence + transient (per ego mode)
#
# Single mode:
#   EGO_MODE=maintain bash analyze/coordination/run_final_mechanism.sh collect
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
cd "$REPO_ROOT"
export PYTHONUNBUFFERED=1

RESULT_10K="${RESULT_10K:-/data/puffer/crosscoder/crosscoder_10k}"
RESULT_ROOT="${RESULT_ROOT:-/data/puffer/crosscoder/crosscoder_mechanism}"
# Historical immutable trees: /data/puffer/results/crosscoder*/

DICT_SIZE=1024
LAMBDA=0.3
ALPHA=0.5
ALPHAS="0.25,0.5,0.75,1.0"

NUM_TRAIN_MAPS="${NUM_TRAIN_MAPS:-10000}"
NUM_VAL_MAPS="${NUM_VAL_MAPS:-10000}"
SHARD_SIZE="${SHARD_SIZE:-1000}"
# maintain | record | reactive | all
EGO_MODE="${EGO_MODE:-maintain}"
DRIVER_SEED="${DRIVER_SEED:-42}"

GPU_ID="${GPU_ID:-0}"
DEVICE="${DEVICE:-cuda}"
FORCE="${FORCE:-0}"
PERSIST_MAPS="${PERSIST_MAPS:-1000}"

have() { [[ -e "$1" ]]; }
skip_msg() { echo "[reuse] $1 (set FORCE=1 to regenerate)"; }
run_py() { CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" "$@"; }

CC="$SCRIPT_DIR/crosscoder"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export FORCE
export RESULT_ROOT RESULT_10K GPU_ID DEVICE

ego_modes() {
  if [[ "$EGO_MODE" == "all" ]]; then
    echo maintain record reactive
  else
    echo "$EGO_MODE"
  fi
}

mech_out() {
  local mode="$1"
  if [[ "$mode" == "maintain" ]]; then
    echo "$RESULT_ROOT"
  else
    echo "${RESULT_ROOT}_ego_${mode}"
  fi
}

acts_dir() {
  local mode="$1"
  if [[ "$mode" == "maintain" ]]; then
    echo "$RESULT_ROOT/policy_seed_replication"
  else
    echo "$RESULT_ROOT/policy_seed_replication/ego_${mode}"
  fi
}

tenk_out() {
  local mode="$1"
  if [[ "$mode" == "maintain" ]]; then
    echo "$RESULT_10K"
  else
    echo "${RESULT_10K}_ego_${mode}"
  fi
}

# ============================================================
help() {
  cat <<'EOF'
ReCord vs Reactive — full mechanistic pipeline

  all         collect+train+replicate+intervene+transient for all 3 ego modes
  collect     EGO_MODE=maintain|record|reactive|all
  crosscoder  confirmatory 10K train (EGO_MODE=all → all three)
  replicate   matched-seed |Δc| + subspace U
  subspace    alias of replicate
  intervene   same-state causal patch
  transient   one-shot persistence + transient characterization

  smoke | check_results | help

  bash analyze/coordination/run_final_mechanism.sh all
  EGO_MODE=maintain bash analyze/coordination/run_final_mechanism.sh intervene

Paths:
  collect  → $RESULT_ROOT/policy_seed_replication/[ego_*]
  10k      → $RESULT_10K[_ego_*]
  mechanism→ $RESULT_ROOT[_ego_*]
EOF
}

# ============================================================
collect() {
  echo "========== collect ego_mode=$EGO_MODE =========="
  local need_run=0
  local m val_shards
  for m in $(ego_modes); do
    val_shards="$(acts_dir "$m")/validation_dataset/shards"
    if [[ "$FORCE" == "1" ]] || ! have "$val_shards"; then
      need_run=1
      break
    fi
    skip_msg "$m activations already present"
  done
  [[ "$need_run" == "1" ]] || return 0

  local force_flag=()
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
crosscoder() {
  if [[ "$EGO_MODE" == "all" ]]; then
    echo "========== crosscoder all ego modes =========="
    RESULT_ROOT="$RESULT_ROOT" RESULT_10K="$RESULT_10K" \
      GPU_ID="$GPU_ID" DEVICE="$DEVICE" FORCE="$FORCE" \
      bash "$CC/run_train_crosscoder_all.sh"
    return 0
  fi
  local out need
  out="$(tenk_out "$EGO_MODE")"
  need="$(acts_dir "$EGO_MODE")/train_dataset/shards"
  echo "========== crosscoder ego_mode=$EGO_MODE → $out =========="
  echo "  frozen: dict=$DICT_SIZE  lambda=$LAMBDA"
  if [[ "$FORCE" != "1" ]] && have "$out/summary.json"; then
    skip_msg "$out/summary.json"
    return 0
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
replicate() {
  if [[ "$EGO_MODE" == "all" ]]; then
    echo "========== replicate all ego modes =========="
    RESULT_ROOT="$RESULT_ROOT" GPU_ID="$GPU_ID" DEVICE="$DEVICE" FORCE="$FORCE" \
      bash "$CC/run_replicate_all_ego.sh"
    return 0
  fi
  local out need
  out="$(mech_out "$EGO_MODE")"
  need="$(acts_dir "$EGO_MODE")/train_dataset/shards"
  echo "========== replicate + subspace ego_mode=$EGO_MODE → $out =========="
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

subspace() { replicate; }

# ============================================================
_intervene_one() {
  local mode="$1"
  local out acts
  out="$(mech_out "$mode")"
  acts="$(acts_dir "$mode")"
  echo "========== intervene ego_mode=$mode out=$out =========="
  echo "  alpha sweep: {$ALPHAS}  primary=$ALPHA"
  if [[ "$FORCE" != "1" ]] && have "$out/intervention/summary.json"; then
    skip_msg "$out/intervention/summary.json"
    return 0
  fi
  if ! have "$out/subspace/consensus"; then
    echo "Need replicate for ego_mode=$mode first ($out/subspace)." >&2
    return 1
  fi
  run_py "$CC/intervention.py" \
    --out-root "$out" \
    --acts-root "$acts" \
    --device "$DEVICE"
}

intervene() {
  local m
  for m in $(ego_modes); do
    _intervene_one "$m"
  done
}

# ============================================================
_transient_one() {
  local mode="$1"
  local out acts
  out="$(mech_out "$mode")"
  acts="$(acts_dir "$mode")"
  echo "========== transient ego_mode=$mode out=$out =========="
  if [[ "$FORCE" == "1" ]] || ! have "$out/intervention_persistence/summary.json"; then
    echo "  [one-shot persistence]"
    run_py "$CC/persistence.py" --out-root "$out" --maps "$PERSIST_MAPS" --device "$DEVICE"
  else
    skip_msg "$out/intervention_persistence/summary.json"
  fi
  if [[ "$FORCE" != "1" ]] && have "$out/transient_causal_characterization/summary.json"; then
    skip_msg "$out/transient_causal_characterization/summary.json"
    return 0
  fi
  run_py "$CC/transient_causal.py" \
    --out-root "$out" \
    --acts-root "$acts" \
    --device "$DEVICE"
}

transient() {
  local m
  for m in $(ego_modes); do
    _transient_one "$m"
  done
}

# ============================================================
smoke() {
  echo "========== smoke =========="
  run_py "$SCRIPT_DIR/crosscoder/test_cleanup_regression.py"
  RESULT_ROOT="$RESULT_ROOT" run_py - <<'PY'
import json
from pathlib import Path
import numpy as np
from crosscoder.intervention import project, load_pair_bases
from crosscoder.frozen_config import HIDDEN_DIM, PRIMARY_ALPHA

root = Path(__import__("os").environ["RESULT_ROOT"])
pairs = json.loads((root / "policy_seed_replication" / "policy_pairs.json").read_text())
pid = pairs["selected_pairs"][0]["pair_id"]
b = load_pair_bases(root, pid)
assert b["U"].shape[0] == HIDDEN_DIM
rng = np.random.default_rng(0)
delta = rng.normal(size=(8, HIDDEN_DIM))
d1 = project(delta, b["U"])
assert np.allclose(project(d1, b["U"]), d1, atol=1e-5)
print(f"PASS bases {pid} U={b['U'].shape} alpha={PRIMARY_ALPHA}")
pers = root / "intervention_persistence" / "summary.json"
if pers.is_file():
    doc = json.loads(pers.read_text())
    hls = [doc["one_shot"][p]["half_life_RU"] for p in doc["one_shot"]]
    print(f"PASS artificial half-lives={hls}")
g6 = root / "transient_causal_characterization" / "summary.json"
if g6.is_file():
    across = json.loads(g6.read_text()).get("across_pairs") or {}
    assert "cum10_positive_mass" in across
    print(f"PASS transient keys={list(across)}")
print("smoke OK")
PY
}

check_results() {
  echo "========== check_results (no rerun) =========="
  RESULT_ROOT="$RESULT_ROOT" RESULT_10K="$RESULT_10K" run_py - <<'PY'
import json, os
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None

root = Path(os.environ["RESULT_ROOT"])
r10 = Path(os.environ["RESULT_10K"])
modes = ["maintain", "record", "reactive"]

print(f"\n[paths] mechanism={root}  10k={r10}")
for mode in modes:
    out = root if mode == "maintain" else Path(f"{root}_ego_{mode}")
    acts = root / "policy_seed_replication" if mode == "maintain" else root / "policy_seed_replication" / f"ego_{mode}"
    t10 = r10 if mode == "maintain" else Path(f"{r10}_ego_{mode}")
    print(f"\n=== ego_mode={mode} ===")
    print(f"  collect shards: {bool((acts/'validation_dataset'/'shards').exists())}")
    print(f"  10k summary:    {bool((t10/'summary.json').exists())}")
    rep = load(out / "policy_seed_replication" / "summary.json") or {}
    across = rep.get("across_pairs") or {}
    dc = across.get("d_c_tight_over_nominal") or {}
    print(f"  |Δc| mean={dc.get('mean')} per_pair={dc.get('per_pair')}")
    sub = load(out / "subspace" / "summary.json") or {}
    print(f"  subspace overlap={sub.get('mean_overlap')}")
    inter = load(out / "intervention" / "summary.json") or {}
    ap = inter.get("across_pairs") or {}
    print(f"  intervene tight@0.5={ap.get('interaction_tight_a0.5')}")
    pers = load(out / "intervention_persistence" / "summary.json") or {}
    if pers.get("one_shot"):
        hls = [pers["one_shot"][p]["half_life_RU"] for p in pers["one_shot"]]
        print(f"  R_U half-life={hls}")
    g6 = load(out / "transient_causal_characterization" / "summary.json") or {}
    print(f"  transient={g6.get('across_pairs')}")
print()
PY
}

# Full matrix: all three ego modes end-to-end
all_stages() {
  echo "========== ALL ego modes end-to-end =========="
  local saved="$EGO_MODE"
  EGO_MODE=all
  collect
  crosscoder
  replicate
  intervene
  transient
  EGO_MODE="$saved"
  echo "========== ALL done =========="
  check_results
}

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
