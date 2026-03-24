#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project"
ABLATION_ROOT="${PROJECT_ROOT}/ablation_workspace"

DATASET="${DATASET:-youtube}"
TASK="${TASK:-STI}"
WANDB_PROJECT="${WANDB_PROJECT:-mmsi-ablation}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
VISUAL_FILM_LAYERS="${VISUAL_FILM_LAYERS:-3}"
FUSION_FILM_LAYERS="${FUSION_FILM_LAYERS:-1}"
CENTER_LOSS_WEIGHT="${CENTER_LOSS_WEIGHT:-0.0}"
SUPCON_WEIGHT="${SUPCON_WEIGHT:-0.0}"

BOTTLENECK_TOKEN_OPTIONS="${BOTTLENECK_TOKEN_OPTIONS:-2 4 8}"
BOTTLENECK_LAYER_OPTIONS="${BOTTLENECK_LAYER_OPTIONS:-1 2}"

cd "${ABLATION_ROOT}"

for bottleneck_tokens in ${BOTTLENECK_TOKEN_OPTIONS}; do
  for bottleneck_layers in ${BOTTLENECK_LAYER_OPTIONS}; do
    model_name="ablation_${TASK,,}_${DATASET}_vf${VISUAL_FILM_LAYERS}_ff${FUSION_FILM_LAYERS}_bottleneck_bt${bottleneck_tokens}_bl${bottleneck_layers}_cl${CENTER_LOSS_WEIGHT}_sc${SUPCON_WEIGHT}"
    echo "Submitting ${model_name}"
    sbatch \
      --job-name="${model_name}" \
      --export=ALL,DATASET="${DATASET}",TASK="${TASK}",USE_WANDB=1,WANDB_PROJECT="${WANDB_PROJECT}",WANDB_ENTITY="${WANDB_ENTITY}",VISUAL_FILM_LAYERS="${VISUAL_FILM_LAYERS}",FUSION_FILM_LAYERS="${FUSION_FILM_LAYERS}",FUSION_MODE="bottleneck",BOTTLENECK_TOKENS="${bottleneck_tokens}",BOTTLENECK_FUSION_LAYERS="${bottleneck_layers}",CENTER_LOSS_WEIGHT="${CENTER_LOSS_WEIGHT}",SUPCON_WEIGHT="${SUPCON_WEIGHT}",MODEL_NAME="${model_name}",WANDB_RUN_NAME="${model_name}" \
      run_ablation_sti.sbatch
  done
done
