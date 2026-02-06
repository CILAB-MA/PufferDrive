#!/usr/bin/env bash
set -euo pipefail
GPU_ID=${1:-0}
CUDA_VISIBLE_DEVICES=$GPU_ID python heatmap.py --mode reactive --type nominal 
CUDA_VISIBLE_DEVICES=$GPU_ID python heatmap.py --mode replay --type nominal 
CUDA_VISIBLE_DEVICES=$GPU_ID python heatmap.py --mode reactive --type lane_breaker 
CUDA_VISIBLE_DEVICES=$GPU_ID python heatmap.py --mode replay --type lane_breaker 
CUDA_VISIBLE_DEVICES=$GPU_ID python heatmap.py --mode reactive --type slow_v2 
CUDA_VISIBLE_DEVICES=$GPU_ID python heatmap.py --mode replay --type slow_v2 
CUDA_VISIBLE_DEVICES=$GPU_ID python heatmap.py --mode reactive --type slow 
CUDA_VISIBLE_DEVICES=$GPU_ID python heatmap.py --mode replay --type slow 
CUDA_VISIBLE_DEVICES=$GPU_ID python heatmap.py --mode reactive --type fast 
CUDA_VISIBLE_DEVICES=$GPU_ID python heatmap.py --mode replay --type fast 