#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project"
ABLATION_ROOT="${PROJECT_ROOT}/ablation_workspace"

DATASET="${DATASET:-youtube}"
TASK="${TASK:-STI}"
WANDB_PROJECT="${WANDB_PROJECT:-mmsi-graph-study}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
SEEDS="${SEEDS:-1234 2234 3234}"

cd "${ABLATION_ROOT}"

for seed in ${SEEDS}; do
  film_model="repeat_${TASK,,}_${DATASET}_vf3_ff1_seed${seed}"
  echo "Submitting ${film_model}"
  sbatch \
    --job-name="${film_model}" \
    --export=ALL,DATASET="${DATASET}",TASK="${TASK}",USE_WANDB=1,WANDB_PROJECT="${WANDB_PROJECT}",WANDB_ENTITY="${WANDB_ENTITY}",VISUAL_FILM_LAYERS="3",FUSION_FILM_LAYERS="1",FUSION_MODE="standard",GRAPH_MODE="standard",GRAPH_LAYERS="0",SEED="${seed}",MODEL_NAME="${film_model}",WANDB_RUN_NAME="${film_model}" \
    run_ablation_sti.sbatch

  graph_model="repeat_${TASK,,}_${DATASET}_vf3_ff1_speaker_centered_gl2_seed${seed}"
  echo "Submitting ${graph_model}"
  sbatch \
    --job-name="${graph_model}" \
    --export=ALL,DATASET="${DATASET}",TASK="${TASK}",USE_WANDB=1,WANDB_PROJECT="${WANDB_PROJECT}",WANDB_ENTITY="${WANDB_ENTITY}",VISUAL_FILM_LAYERS="3",FUSION_FILM_LAYERS="1",FUSION_MODE="standard",GRAPH_MODE="speaker_centered",GRAPH_LAYERS="2",SEED="${seed}",MODEL_NAME="${graph_model}",WANDB_RUN_NAME="${graph_model}" \
    run_ablation_sti.sbatch
done
