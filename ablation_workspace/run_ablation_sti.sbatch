#!/usr/bin/env bash
#SBATCH --job-name=mmsi-ablation
#SBATCH --account=gpu-free
#SBATCH --qos=gpu-free
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project/ablation_workspace/checkpoints/slurm-%j.out

set -euo pipefail

cd /mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project/ablation_workspace
mkdir -p checkpoints

export DATASET="${DATASET:-youtube}"
export TASK="${TASK:-STI}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-mmsi-ablation}"
export WANDB_ENTITY="${WANDB_ENTITY:-gloriaa_zxy-university-of-manchester}"
export VISUAL_FILM_LAYERS="${VISUAL_FILM_LAYERS:-0}"
export FUSION_FILM_LAYERS="${FUSION_FILM_LAYERS:-0}"
export CENTER_LOSS_WEIGHT="${CENTER_LOSS_WEIGHT:-0.0}"
export MODEL_NAME="${MODEL_NAME:-ablation_${TASK,,}_${DATASET}_vf${VISUAL_FILM_LAYERS}_ff${FUSION_FILM_LAYERS}_cl${CENTER_LOSS_WEIGHT}_job${SLURM_JOB_ID}}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${MODEL_NAME}}"

./run_ablation_sti.sh
