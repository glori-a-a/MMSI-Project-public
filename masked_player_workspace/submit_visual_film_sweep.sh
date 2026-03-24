#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-youtube}"
TASK="${TASK:-STI}"
WANDB_PROJECT="${WANDB_PROJECT:-mmsi-ablation}"

for visual_layers in 1 2 3 4; do
  sbatch \
    --export=ALL,DATASET="${DATASET}",TASK="${TASK}",WANDB_PROJECT="${WANDB_PROJECT}",VISUAL_FILM_LAYERS="${visual_layers}",FUSION_FILM_LAYERS=0,CENTER_LOSS_WEIGHT=0.0 \
    /mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project/ablation_workspace/run_ablation_sti.sbatch
done
