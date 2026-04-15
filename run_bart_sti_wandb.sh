#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project-ablation"
MMSI_ROOT="${PROJECT_ROOT}/MMSI"
DATASET="${DATASET:-youtube}"
WANDB_PROJECT="${WANDB_PROJECT:-mmsi-bart}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-bart_sti_${DATASET}}"

DATA_ROOT="${PROJECT_ROOT}/data/mmsi"
BENCHMARK_ROOT="${DATA_ROOT}/benchmark"
KEYPOINT_ROOT="${DATA_ROOT}/keypoints"

if [[ "${DATASET}" == "youtube" ]]; then
  TXT_DIR="${BENCHMARK_ROOT}/youtube/transcripts/anonymized"
  TXT_LABELED_DIR="${BENCHMARK_ROOT}/youtube/transcripts/anonymized_labeled"
  KEYPOINT_DIR="${KEYPOINT_ROOT}/keypoints_youtube"
  META_DIR="${BENCHMARK_ROOT}/youtube/meta_data"
  DATA_SPLIT_FILE="${BENCHMARK_ROOT}/youtube/data_split.json"
  CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/bart_sti_youtube"
else
  TXT_DIR="${BENCHMARK_ROOT}/ego4d/transcripts/anonymized"
  TXT_LABELED_DIR="${BENCHMARK_ROOT}/ego4d/transcripts/anonymized_labeled"
  KEYPOINT_DIR="${KEYPOINT_ROOT}/keypoints_ego4d"
  META_DIR="${BENCHMARK_ROOT}/ego4d/meta_data"
  DATA_SPLIT_FILE="${BENCHMARK_ROOT}/ego4d/data_split.json"
  CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/bart_sti_ego4d"
fi

cd "${PROJECT_ROOT}"
source .venv/bin/activate
cd "${MMSI_ROOT}"

python train.py \
  --model_name "bart_sti_${DATASET}" \
  --task "STI" \
  --txt_dir "${TXT_DIR}" \
  --txt_labeled_dir "${TXT_LABELED_DIR}" \
  --keypoint_dir "${KEYPOINT_DIR}" \
  --meta_dir "${META_DIR}" \
  --data_split_file "${DATA_SPLIT_FILE}" \
  --checkpoint_save_dir "${CHECKPOINT_DIR}" \
  --language_model "bart" \
  --text_pooling "mask" \
  --visual_feature_type "keypoint" \
  --max_people_num 6 \
  --context_length 5 \
  --batch_size 16 \
  --learning_rate 5e-6 \
  --epochs 50 \
  --epochs_warmup 10 \
  --use_wandb \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_run_name "${WANDB_RUN_NAME}" \
  ${WANDB_ENTITY:+--wandb_entity "${WANDB_ENTITY}"}
