#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project"
ABLATION_ROOT="${PROJECT_ROOT}/ablation_workspace"
MMSI_ROOT="${ABLATION_ROOT}/MMSI"

DATASET="${DATASET:-youtube}"
TASK="${TASK:-STI}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-mmsi-ablation}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
VISUAL_FILM_LAYERS="${VISUAL_FILM_LAYERS:-0}"
FUSION_FILM_LAYERS="${FUSION_FILM_LAYERS:-0}"
FUSION_MODE="${FUSION_MODE:-standard}"
BOTTLENECK_TOKENS="${BOTTLENECK_TOKENS:-4}"
BOTTLENECK_FUSION_LAYERS="${BOTTLENECK_FUSION_LAYERS:-1}"
CENTER_LOSS_WEIGHT="${CENTER_LOSS_WEIGHT:-0.0}"
SUPCON_WEIGHT="${SUPCON_WEIGHT:-0.0}"
SUPCON_TEMPERATURE="${SUPCON_TEMPERATURE:-0.07}"
MODEL_NAME="${MODEL_NAME:-ablation_${TASK,,}_${DATASET}_vf${VISUAL_FILM_LAYERS}_ff${FUSION_FILM_LAYERS}_${FUSION_MODE}_bt${BOTTLENECK_TOKENS}_bl${BOTTLENECK_FUSION_LAYERS}_cl${CENTER_LOSS_WEIGHT}_sc${SUPCON_WEIGHT}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${MODEL_NAME}}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/mmsi}"
BENCHMARK_ROOT="${DATA_ROOT}/benchmark"
KEYPOINT_ROOT="${DATA_ROOT}/keypoints"

if [[ "${DATASET}" == "youtube" ]]; then
  TXT_DIR="${BENCHMARK_ROOT}/youtube/transcripts/anonymized"
  TXT_LABELED_DIR="${BENCHMARK_ROOT}/youtube/transcripts/anonymized_labeled"
  KEYPOINT_DIR="${KEYPOINT_ROOT}/keypoints_youtube"
  META_DIR="${BENCHMARK_ROOT}/youtube/meta_data"
  DATA_SPLIT_FILE="${BENCHMARK_ROOT}/youtube/data_split.json"
elif [[ "${DATASET}" == "ego4d" ]]; then
  TXT_DIR="${BENCHMARK_ROOT}/ego4d/transcripts/anonymized"
  TXT_LABELED_DIR="${BENCHMARK_ROOT}/ego4d/transcripts/anonymized_labeled"
  KEYPOINT_DIR="${KEYPOINT_ROOT}/keypoints_ego4d"
  META_DIR="${BENCHMARK_ROOT}/ego4d/meta_data"
  DATA_SPLIT_FILE="${BENCHMARK_ROOT}/ego4d/data_split.json"
else
  echo "Unsupported DATASET='${DATASET}'. Use DATASET=youtube or DATASET=ego4d."
  exit 1
fi

CHECKPOINT_DIR="${ABLATION_ROOT}/checkpoints/${MODEL_NAME}"

cd "${PROJECT_ROOT}"
source .venv/bin/activate

python - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available in current torch.")
    sys.exit(1)
try:
    torch.cuda.init()
    _ = torch.zeros(1, device="cuda")
except Exception as exc:
    print("ERROR: CUDA runtime check failed.")
    print(exc)
    sys.exit(1)
print(f"CUDA ready. torch={torch.__version__}, cuda={torch.version.cuda}")
PY

cd "${MMSI_ROOT}"

WANDB_ARGS=()
if [[ "${USE_WANDB}" == "1" ]]; then
  WANDB_ARGS+=(--use_wandb --wandb_project "${WANDB_PROJECT}" --wandb_run_name "${WANDB_RUN_NAME}")
  if [[ -n "${WANDB_ENTITY}" ]]; then
    WANDB_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
  fi
fi

python train.py \
  --model_name "${MODEL_NAME}" \
  --task "${TASK}" \
  --txt_dir "${TXT_DIR}" \
  --txt_labeled_dir "${TXT_LABELED_DIR}" \
  --keypoint_dir "${KEYPOINT_DIR}" \
  --meta_dir "${META_DIR}" \
  --data_split_file "${DATA_SPLIT_FILE}" \
  --checkpoint_save_dir "${CHECKPOINT_DIR}" \
  --language_model "bert" \
  --max_people_num 6 \
  --context_length 5 \
  --batch_size 16 \
  --learning_rate 5e-6 \
  --epochs 200 \
  --epochs_warmup 10 \
  --visual_film_layers "${VISUAL_FILM_LAYERS}" \
  --fusion_film_layers "${FUSION_FILM_LAYERS}" \
  --fusion_mode "${FUSION_MODE}" \
  --bottleneck_tokens "${BOTTLENECK_TOKENS}" \
  --bottleneck_fusion_layers "${BOTTLENECK_FUSION_LAYERS}" \
  --center_loss_weight "${CENTER_LOSS_WEIGHT}" \
  --supcon_weight "${SUPCON_WEIGHT}" \
  --supcon_temperature "${SUPCON_TEMPERATURE}" \
  "${WANDB_ARGS[@]}"
