#!/usr/bin/env bash
GPU_ID=${1:-0}
MODELS=()

for f in /data/puffer/experiments/nominal/puffer_drive_*.pt; do
  bn=$(basename "$f") # puffer_drive_xxx.pt
  id=${bn#puffer_drive_} # xxx.pt
  id=${id%.pt} # xxx
  MODELS+=("$id")
done
echo "Found models: ${MODELS[*]}"

# for MP in "${MODELS[@]}"; do

#   echo "Running reactive-play: ${MP} vs ${MP}"
#   CUDA_VISIBLE_DEVICES=$GPU_ID puffer zeroshot puffer_drive \
#     --load-multiple-model-path "/data/puffer/experiments/nominal/puffer_drive_${MP}.pt" \
#                                "/data/puffer/experiments/nominal/puffer_drive_${MP}.pt" \
#     --zero-shot-mode "save-replay"
# done

for MP1 in "${MODELS[@]}"; do
  for MP2 in "${MODELS[@]}"; do
    # echo "Running replay: ${MP1} vs ${MP2}"
    # CUDA_VISIBLE_DEVICES=$GPU_ID puffer zeroshot puffer_drive \
    #   --load-multiple-model-path "/data/puffer/experiments/nominal/puffer_drive_${MP1}.pt" \
    #                              "/data/puffer/experiments/nominal/puffer_drive_${MP2}.pt" \
    #   --zero-shot-mode "replay"
    echo "Running reactive-play: ${MP1} vs ${MP2}"
    CUDA_VISIBLE_DEVICES=$GPU_ID puffer zeroshot puffer_drive \
      --load-multiple-model-path "/data/puffer/experiments/nominal/puffer_drive_${MP1}.pt" \
                                 "/data/puffer/experiments/nominal/puffer_drive_${MP2}.pt" \
      --zero-shot-mode "reactive-play"
  done
done