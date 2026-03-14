#!/usr/bin/env bash
#SBATCH --job-name=mmsi-sti-wandb
#SBATCH --account=gpu-free
#SBATCH --qos=gpu-free
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project/checkpoints/slurm-%j.out

set -euo pipefail

cd /mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project
source .venv/bin/activate

# Optional: set HF token to avoid unauthenticated Hub warnings/rate limits.
# export HF_TOKEN="your_token_here"

export DATASET="${DATASET:-youtube}"
export USE_WANDB=1
export WANDB_PROJECT="${WANDB_PROJECT:-mmsi-baseline}"
export WANDB_ENTITY="${WANDB_ENTITY:-gloriaa_zxy-university-of-manchester}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-baseline_sti_bert_${DATASET}_job${SLURM_JOB_ID}}"

./run_baseline_sti.sh
