#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project-ablation"
MMSI_ROOT="${PROJECT_ROOT}/MMSI"

RAW_VIDEO_DIR="${PROJECT_ROOT}/data/raw_video_sources/werewolf_among_us/Youtube/videos_mmsi_named"
TRANSCRIPT_DIR="${PROJECT_ROOT}/data/mmsi/benchmark/youtube/transcripts/anonymized"
MARLIN_CACHE_DIR="${PROJECT_ROOT}/data/marlin_features/youtube_medium_utt_v3"
MARLIN_CHECKPOINT="${PROJECT_ROOT}/.marlin/marlin_vit_base_ytf.encoder.pt"

cd "${PROJECT_ROOT}"
source .venv/bin/activate
cd "${MMSI_ROOT}"

python extract_utt_features.py \
  --video_dir "${RAW_VIDEO_DIR}" \
  --transcript_dir "${TRANSCRIPT_DIR}" \
  --out_dir "${MARLIN_CACHE_DIR}" \
  --encoder marlin \
  --marlin_checkpoint "${MARLIN_CHECKPOINT}" \
  --sequence_length 16 \
  --video_fps 5 \
  --frame_size 224 224 \
  --device cuda
