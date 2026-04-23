#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project-ablation"
MMSI_ROOT="${PROJECT_ROOT}/MMSI"
WANDB_PROJECT="${WANDB_PROJECT:-Roberta+vit}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-roberta_sti_keypoint_cached_vit_youtube_medium_utt_v3}"

DATA_ROOT="${PROJECT_ROOT}/data/mmsi"
BENCHMARK_ROOT="${DATA_ROOT}/benchmark"
KEYPOINT_ROOT="${DATA_ROOT}/keypoints"
VIT_CACHE_DIR="${PROJECT_ROOT}/data/vit_features/youtube_medium_utt_v3"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
cd "${MMSI_ROOT}"

python train.py \
  --model_name "roberta_sti_keypoint_cached_vit_youtube_medium_utt_v3" \
  --task "STI" \
  --txt_dir "${BENCHMARK_ROOT}/youtube/transcripts/anonymized" \
  --txt_labeled_dir "${BENCHMARK_ROOT}/youtube/transcripts/anonymized_labeled" \
  --keypoint_dir "${KEYPOINT_ROOT}/keypoints_youtube" \
  --meta_dir "${BENCHMARK_ROOT}/youtube/meta_data" \
  --data_split_file "${BENCHMARK_ROOT}/youtube/data_split.json" \
  --video_dir "${VIT_CACHE_DIR}" \
  --checkpoint_save_dir "${PROJECT_ROOT}/checkpoints/roberta_sti_keypoint_cached_vit_youtube_medium_utt_v3" \
  --language_model "roberta" \
  --text_pooling "mask" \
  --visual_feature_type "keypoint_vit" \
  --precomputed_visual_features \
  --max_people_num 6 \
  --context_length 5 \
  --batch_size 16 \
  --learning_rate 5e-6 \
  --epochs 40 \
  --epochs_warmup 2 \
  --workers 1 \
  --max_train_samples 1024 \
  --max_test_samples 512 \
  --use_wandb \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  ${WANDB_ENTITY:+--wandb_entity "${WANDB_ENTITY}"}
