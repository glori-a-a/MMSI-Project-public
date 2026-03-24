#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project"
ABLATION_ROOT="${PROJECT_ROOT}/ablation_workspace"

DATASET="${DATASET:-youtube}"
TASK="${TASK:-STI}"
WANDB_PROJECT="${WANDB_PROJECT:-mmsi-graph-study}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
GRAPH_LAYER_OPTIONS="${GRAPH_LAYER_OPTIONS:-1 2 3}"
SEED="${SEED:-1234}"

cd "${ABLATION_ROOT}"

for graph_layers in ${GRAPH_LAYER_OPTIONS}; do
  model_name="graphv3_${TASK,,}_${DATASET}_vf3_ff1_speaker_contextual_gl${graph_layers}_seed${SEED}"
  echo "Submitting ${model_name}"
  sbatch \
    --job-name="${model_name}" \
    --export=ALL,DATASET="${DATASET}",TASK="${TASK}",USE_WANDB=1,WANDB_PROJECT="${WANDB_PROJECT}",WANDB_ENTITY="${WANDB_ENTITY}",VISUAL_FILM_LAYERS="3",FUSION_FILM_LAYERS="1",FUSION_MODE="standard",GRAPH_MODE="speaker_contextual",GRAPH_LAYERS="${graph_layers}",SEED="${SEED}",MODEL_NAME="${model_name}",WANDB_RUN_NAME="${model_name}" \
    run_ablation_sti.sbatch
done
