#!/usr/bin/env bash
GPU_ID=${1:-0}
MODE=${2:-nominal}

CUDA_VISIBLE_DEVICES=$GPU_ID puffer zeroshot puffer_drive_pbt