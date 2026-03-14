#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-youtube}"
TASK="${TASK:-STI}"
WANDB_PROJECT="${WANDB_PROJECT:-mmsi-ablation}"
BEST_VISUAL_FILM_LAYERS="${BEST_VISUAL_FILM_LAYERS:-1}"

for fusion_layers in 1 2 3 4; do
  sbatch \
    --export=ALL,DATASET="${DATASET}",TASK="${TASK}",WANDB_PROJECT="${WANDB_PROJECT}",VISUAL_FILM_LAYERS="${BEST_VISUAL_FILM_LAYERS}",FUSION_FILM_LAYERS="${fusion_layers}",CENTER_LOSS_WEIGHT=0.0 \
    /mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project/ablation_workspace/run_ablation_sti.sbatch
done
