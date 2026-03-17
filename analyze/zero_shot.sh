#!/usr/bin/env bash
GPU_ID=${1:-0}
MODE=${2:-unseen_other_seeds} # or unseen_other_rewards
FOLDER=${3:-reactive} # or replay, selfplay
POPULATION_MODE=${4:-popul_lane_nominal} # population folder
EGOS=()
OTHERS=()

for f in /data/puffer/experiments/${FOLDER}/puffer_drive_*.pt; do
  bn=$(basename "$f") # puffer_drive_xxx.pt
  id=${bn#puffer_drive_} # xxx.pt
  id=${id%.pt} # xxx
  EGOS+=("$id")
done

for f in /data/puffer/${POPULATION_MODE}/${MODE}/puffer_drive_*.pt; do
  bn=$(basename "$f") # puffer_drive_xxx.pt
  id=${bn#puffer_drive_} # xxx.pt
  id=${id%.pt} # xxx
  OTHERS+=("$id")
done

echo "Found models: ${EGOS[*]} ${OTHERS[*]}"

for MP1 in "${EGOS[@]}"; do
  for MP2 in "${OTHERS[@]}"; do
    echo "Running reactive-play: ${MP1} vs ${MP2}"
    CUDA_VISIBLE_DEVICES=$GPU_ID puffer zeroshot puffer_drive \
      --load-multiple-model-path "/data/puffer/experiments/${FOLDER}/puffer_drive_${MP1}.pt" \
                                 "/data/puffer/${POPULATION_MODE}/${MODE}/puffer_drive_${MP2}.pt" \
      --zero-shot-mode "reactive-play" --env.termination-mode "0" 
  done
done