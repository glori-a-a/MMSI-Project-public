#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project-ablation"
MMSI_ROOT="${PROJECT_ROOT}/MMSI"
WANDB_PROJECT="${WANDB_PROJECT:-mmsi-bart-cached-marlin-medium}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-bart_sti_keypoint_cached_marlin_youtube_medium}"

DATA_ROOT="${PROJECT_ROOT}/data/mmsi"
BENCHMARK_ROOT="${DATA_ROOT}/benchmark"
KEYPOINT_ROOT="${DATA_ROOT}/keypoints"
MARLIN_CACHE_DIR="${PROJECT_ROOT}/data/marlin_features/youtube_medium"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
cd "${MMSI_ROOT}"

python train.py \
  --model_name "bart_sti_keypoint_cached_marlin_youtube_medium" \
  --task "STI" \
  --txt_dir "${BENCHMARK_ROOT}/youtube/transcripts/anonymized" \
  --txt_labeled_dir "${BENCHMARK_ROOT}/youtube/transcripts/anonymized_labeled" \
  --keypoint_dir "${KEYPOINT_ROOT}/keypoints_youtube" \
  --meta_dir "${BENCHMARK_ROOT}/youtube/meta_data" \
  --data_split_file "${BENCHMARK_ROOT}/youtube/data_split.json" \
  --video_dir "${MARLIN_CACHE_DIR}" \
  --checkpoint_save_dir "${PROJECT_ROOT}/checkpoints/bart_sti_keypoint_cached_marlin_youtube_medium" \
  --language_model "bart" \
  --text_pooling "mask" \
  --visual_feature_type "keypoint_marlin" \
  --precomputed_visual_features \
  --max_people_num 6 \
  --context_length 5 \
  --batch_size 16 \
  --learning_rate 5e-6 \
  --epochs 30 \
  --epochs_warmup 2 \
  --workers 1 \
  --max_train_samples 512 \
  --max_test_samples 256 \
  --use_wandb \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  ${WANDB_ENTITY:+--wandb_entity "${WANDB_ENTITY}"}
