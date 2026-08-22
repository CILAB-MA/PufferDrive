from __future__ import annotations

from pathlib import Path

# --- Crosscoder (confirmatory freeze) ---
FROZEN_DICT = 1024
FROZEN_LAMBDA = 0.3
FROZEN_K = 32  # U ∈ R^{256×32}
HIDDEN_DIM = 256

# --- Causal intervention ---
PRIMARY_ALPHA = 0.5
ALPHAS = (0.25, 0.5, 0.75, 1.0)

# --- Transient event alignment ---
EVENT_WINDOW_LO = -20
EVENT_WINDOW_HI = 20  # inclusive via arange(lo, hi+1)

# --- Policy matching ---
SEED_ORDER = (42, 3, 11, 0)
EXPERIMENTS_ROOT = Path("/data/puffer/experiments")
REC_DIR = EXPERIMENTS_ROOT / "record-uniform-popul_lane_nominal-wandb"
REA_DIR = EXPERIMENTS_ROOT / "reactive-uniform-popul_lane_nominal-wandb"

# --- Active result roots ---
RESULTS_BASE = Path("/data/puffer/crosscoder")
RESULTS_10K = RESULTS_BASE / "crosscoder_10k"
RESULTS_MECHANISM = RESULTS_BASE / "crosscoder_mechanism"
SPLIT_PATH = RESULTS_10K / "config" / "train_dev_split.json"
# Historical immutable trees (read-only reference):
#   /data/puffer/results/crosscoder{,_10k,_mechanism}/
