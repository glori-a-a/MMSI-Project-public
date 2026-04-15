#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project-ablation"
MMSI_ROOT="${PROJECT_ROOT}/MMSI"
WANDB_PROJECT="${WANDB_PROJECT:-mmsi-roberta}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-roberta_sti_keypoint_youtube}"

DATA_ROOT="${PROJECT_ROOT}/data/mmsi"
BENCHMARK_ROOT="${DATA_ROOT}/benchmark"
KEYPOINT_ROOT="${DATA_ROOT}/keypoints"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
cd "${MMSI_ROOT}"

python train.py \
  --model_name "roberta_sti_keypoint_youtube" \
  --task "STI" \
  --txt_dir "${BENCHMARK_ROOT}/youtube/transcripts/anonymized" \
  --txt_labeled_dir "${BENCHMARK_ROOT}/youtube/transcripts/anonymized_labeled" \
  --keypoint_dir "${KEYPOINT_ROOT}/keypoints_youtube" \
  --meta_dir "${BENCHMARK_ROOT}/youtube/meta_data" \
  --data_split_file "${BENCHMARK_ROOT}/youtube/data_split.json" \
  --checkpoint_save_dir "${PROJECT_ROOT}/checkpoints/roberta_sti_keypoint_youtube" \
  --language_model "roberta" \
  --text_pooling "mask" \
  --visual_feature_type "keypoint" \
  --max_people_num 6 \
  --context_length 5 \
  --batch_size 16 \
  --learning_rate 5e-6 \
  --epochs 80 \
  --epochs_warmup 10 \
  --workers 1 \
  --use_wandb \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  ${WANDB_ENTITY:+--wandb_entity "${WANDB_ENTITY}"}
