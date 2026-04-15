#!/usr/bin/env bash
#SBATCH --job-name=mmsi-bart-cvitl
#SBATCH --account=gpu-free
#SBATCH --qos=gpu-free
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project-ablation/checkpoints/slurm-%j.out

set -euo pipefail

cd /mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project-ablation
bash ./run_extract_vit_cache_youtube_large.sh
bash ./run_bart_sti_cached_vit_large_wandb.sh
