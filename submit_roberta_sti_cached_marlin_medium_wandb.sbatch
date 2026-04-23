#!/usr/bin/env bash
#SBATCH --job-name=mmsi-rbt-cmarm
#SBATCH --account=gpu-free
#SBATCH --qos=gpu-free
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project-ablation/checkpoints/slurm-%j.out

set -euo pipefail

cd /mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project-ablation
bash ./run_extract_utt_marlin_cache_youtube_medium.sh
bash ./run_roberta_sti_cached_marlin_medium_wandb.sh
