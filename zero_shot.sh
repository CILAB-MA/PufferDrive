#!/usr/bin/env bash
GPU_ID=${1:-0}
MODE=${2:-nominal}
EGOS=()
OTHERS=()

for f in /data/puffer/experiments/nominal/puffer_drive_*.pt; do
  bn=$(basename "$f") # puffer_drive_xxx.pt
  id=${bn#puffer_drive_} # xxx.pt
  id=${id%.pt} # xxx
  EGOS+=("$id")
done

for f in /data/puffer/experiments/${MODE}/puffer_drive_*.pt; do
  bn=$(basename "$f") # puffer_drive_xxx.pt
  id=${bn#puffer_drive_} # xxx.pt
  id=${id%.pt} # xxx
  OTHERS+=("$id")
done

echo "Found models: ${EGOS[*]} ${OTHERS[*]}"

for OTHER in "${OTHERS[@]}"; do

  echo "Running reactive-play: OTHER MODE ${MODE} ${OTHER} vs ${OTHER}"
  CUDA_VISIBLE_DEVICES=$GPU_ID puffer zeroshot puffer_drive \
    --load-multiple-model-path "/data/puffer/experiments/${MODE}/puffer_drive_${OTHER}.pt" \
                               "/data/puffer/experiments/${MODE}/puffer_drive_${OTHER}.pt" \
    --zero-shot-mode "save-replay"
done

for MP1 in "${EGOS[@]}"; do
  for MP2 in "${OTHERS[@]}"; do
    echo "Running replay: ${MP1} vs ${MP2}"
    CUDA_VISIBLE_DEVICES=$GPU_ID puffer zeroshot puffer_drive \
      --load-multiple-model-path "/data/puffer/experiments/nominal/puffer_drive_${MP1}.pt" \
                                 "/data/puffer/experiments/${MODE}/puffer_drive_${MP2}.pt" \
      --zero-shot-mode "replay"
    echo "Running reactive-play: ${MP1} vs ${MP2}"
    CUDA_VISIBLE_DEVICES=$GPU_ID puffer zeroshot puffer_drive \
      --load-multiple-model-path "/data/puffer/experiments/nominal/puffer_drive_${MP1}.pt" \
                                 "/data/puffer/experiments/${MODE}/puffer_drive_${MP2}.pt" \
      --zero-shot-mode "reactive-play"
  done
done