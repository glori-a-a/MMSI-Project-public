#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project-ablation"
MMSI_ROOT="${PROJECT_ROOT}/MMSI"
WANDB_PROJECT="${WANDB_PROJECT:-mmsi-gpt2-video-quick}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-gpt2_sti_keypoint_vit_youtube_quick}"

DATA_ROOT="${PROJECT_ROOT}/data/mmsi"
BENCHMARK_ROOT="${DATA_ROOT}/benchmark"
KEYPOINT_ROOT="${DATA_ROOT}/keypoints"
RAW_VIDEO_ROOT="${PROJECT_ROOT}/data/raw_video_sources/werewolf_among_us"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
cd "${MMSI_ROOT}"

python train.py \
  --model_name "gpt2_sti_keypoint_vit_youtube_quick" \
  --task "STI" \
  --txt_dir "${BENCHMARK_ROOT}/youtube/transcripts/anonymized" \
  --txt_labeled_dir "${BENCHMARK_ROOT}/youtube/transcripts/anonymized_labeled" \
  --keypoint_dir "${KEYPOINT_ROOT}/keypoints_youtube" \
  --meta_dir "${BENCHMARK_ROOT}/youtube/meta_data" \
  --data_split_file "${BENCHMARK_ROOT}/youtube/data_split.json" \
  --video_dir "${RAW_VIDEO_ROOT}/Youtube/videos_mmsi_named" \
  --checkpoint_save_dir "${PROJECT_ROOT}/checkpoints/gpt2_sti_keypoint_vit_youtube_quick" \
  --language_model "gpt2" \
  --text_pooling "last" \
  --visual_feature_type "keypoint_vit" \
  --max_people_num 6 \
  --context_length 5 \
  --batch_size 8 \
  --learning_rate 5e-6 \
  --epochs 1 \
  --epochs_warmup 0 \
  --workers 4 \
  --max_train_samples 256 \
  --max_test_samples 128 \
  --use_wandb \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  ${WANDB_ENTITY:+--wandb_entity "${WANDB_ENTITY}"}
