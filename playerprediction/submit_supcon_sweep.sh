#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-youtube}"
TASK="${TASK:-STI}"
WANDB_PROJECT="${WANDB_PROJECT:-mmsi-ablation}"
BEST_VISUAL_FILM_LAYERS="${BEST_VISUAL_FILM_LAYERS:-3}"
BEST_FUSION_FILM_LAYERS="${BEST_FUSION_FILM_LAYERS:-1}"

for supcon_weight in 0.05 0.1 0.2 0.3; do
  sbatch \
    --export=ALL,DATASET="${DATASET}",TASK="${TASK}",WANDB_PROJECT="${WANDB_PROJECT}",VISUAL_FILM_LAYERS="${BEST_VISUAL_FILM_LAYERS}",FUSION_FILM_LAYERS="${BEST_FUSION_FILM_LAYERS}",CENTER_LOSS_WEIGHT=0.0,SUPCON_WEIGHT="${supcon_weight}" \
    /mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project/ablation_workspace/run_ablation_sti.sbatch
done
