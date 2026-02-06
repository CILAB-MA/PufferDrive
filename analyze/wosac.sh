#!/usr/bin/env bash
GPU_ID=${1:-0}
MODE=${2:-nominal}
EGOS=()
OTHERS=()

for f in /data/puffer/experiments/${MODE}/puffer_drive_*.pt; do
  bn=$(basename "$f") # puffer_drive_xxx.pt
  id=${bn#puffer_drive_} # xxx.pt
  id=${id%.pt} # xxx
  EGOS+=("$id")
done

echo "Found models: ${EGOS[*]}"

for EGO in "${EGOS[@]}"; do
  echo "Running log-replay: EGO ${EGO} vs ${EGO}"
  CUDA_VISIBLE_DEVICES=$GPU_ID puffer eval puffer_drive --eval.wosac-realism-eval True --load-model-path "/data/puffer/experiments/${MODE}/puffer_drive_${EGO}.pt"
done
