#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-youtube}"
TASK="${TASK:-STI}"
WANDB_PROJECT="${WANDB_PROJECT:-mmsi-ablation}"
BEST_VISUAL_FILM_LAYERS="${BEST_VISUAL_FILM_LAYERS:-1}"
BEST_FUSION_FILM_LAYERS="${BEST_FUSION_FILM_LAYERS:-1}"

for center_weight in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
  sbatch \
    --export=ALL,DATASET="${DATASET}",TASK="${TASK}",WANDB_PROJECT="${WANDB_PROJECT}",VISUAL_FILM_LAYERS="${BEST_VISUAL_FILM_LAYERS}",FUSION_FILM_LAYERS="${BEST_FUSION_FILM_LAYERS}",CENTER_LOSS_WEIGHT="${center_weight}" \
    /mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project/ablation_workspace/run_ablation_sti.sbatch
done
