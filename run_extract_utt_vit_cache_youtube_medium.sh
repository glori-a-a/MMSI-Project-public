#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project-ablation"
MMSI_ROOT="${PROJECT_ROOT}/MMSI"

RAW_VIDEO_DIR="${PROJECT_ROOT}/data/raw_video_sources/werewolf_among_us/Youtube/videos_mmsi_named"
TRANSCRIPT_DIR="${PROJECT_ROOT}/data/mmsi/benchmark/youtube/transcripts/anonymized"
VIT_CACHE_DIR="${PROJECT_ROOT}/data/vit_features/youtube_medium_utt_v3"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
cd "${MMSI_ROOT}"

python extract_utt_features.py \
  --video_dir "${RAW_VIDEO_DIR}" \
  --transcript_dir "${TRANSCRIPT_DIR}" \
  --out_dir "${VIT_CACHE_DIR}" \
  --encoder vit \
  --sequence_length 16 \
  --video_fps 5 \
  --frame_size 224 224 \
  --device cuda
