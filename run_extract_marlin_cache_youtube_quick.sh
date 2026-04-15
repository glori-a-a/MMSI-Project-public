#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project-ablation"
MMSI_ROOT="${PROJECT_ROOT}/MMSI"

DATA_ROOT="${PROJECT_ROOT}/data/mmsi"
BENCHMARK_ROOT="${DATA_ROOT}/benchmark"
KEYPOINT_ROOT="${DATA_ROOT}/keypoints"
RAW_VIDEO_ROOT="${PROJECT_ROOT}/data/raw_video_sources/werewolf_among_us"
MARLIN_CACHE_DIR="${PROJECT_ROOT}/data/marlin_features/youtube_quick"
MARLIN_CHECKPOINT="${PROJECT_ROOT}/.marlin/marlin_vit_base_ytf.encoder.pt"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
cd "${MMSI_ROOT}"

python extract_marlin_features.py \
  --task "STI" \
  --txt_dir "${BENCHMARK_ROOT}/youtube/transcripts/anonymized" \
  --txt_labeled_dir "${BENCHMARK_ROOT}/youtube/transcripts/anonymized_labeled" \
  --keypoint_dir "${KEYPOINT_ROOT}/keypoints_youtube" \
  --meta_dir "${BENCHMARK_ROOT}/youtube/meta_data" \
  --data_split_file "${BENCHMARK_ROOT}/youtube/data_split.json" \
  --video_dir "${RAW_VIDEO_ROOT}/Youtube/videos_mmsi_named" \
  --output_dir "${MARLIN_CACHE_DIR}" \
  --marlin_checkpoint "${MARLIN_CHECKPOINT}" \
  --language_model "gpt2" \
  --context_length 5 \
  --batch_size 4 \
  --workers 1 \
  --max_train_samples 128 \
  --max_test_samples 64
