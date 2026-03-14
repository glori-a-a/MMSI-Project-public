#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/iusers01/fse-ugpgt01/eee01/t66389xz/MMSI-Project"
MMSI_ROOT="${PROJECT_ROOT}/MMSI"

# Choose dataset: youtube or ego4d
DATASET="${DATASET:-youtube}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-mmsi-baseline}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-baseline_sti_bert_${DATASET}}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/mmsi}"
BENCHMARK_ROOT="${DATA_ROOT}/benchmark"
KEYPOINT_ROOT="${DATA_ROOT}/keypoints"

if [[ "${DATASET}" == "youtube" ]]; then
  TXT_DIR="${BENCHMARK_ROOT}/youtube/transcripts/anonymized"
  TXT_LABELED_DIR="${BENCHMARK_ROOT}/youtube/transcripts/anonymized_labeled"
  KEYPOINT_DIR="${KEYPOINT_ROOT}/keypoints_youtube"
  META_DIR="${BENCHMARK_ROOT}/youtube/meta_data"
  DATA_SPLIT_FILE="${BENCHMARK_ROOT}/youtube/data_split.json"
  CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/sti_bert_youtube"
elif [[ "${DATASET}" == "ego4d" ]]; then
  TXT_DIR="${BENCHMARK_ROOT}/ego4d/transcripts/anonymized"
  TXT_LABELED_DIR="${BENCHMARK_ROOT}/ego4d/transcripts/anonymized_labeled"
  KEYPOINT_DIR="${KEYPOINT_ROOT}/keypoints_ego4d"
  META_DIR="${BENCHMARK_ROOT}/ego4d/meta_data"
  DATA_SPLIT_FILE="${BENCHMARK_ROOT}/ego4d/data_split.json"
  CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/sti_bert_ego4d"
else
  echo "Unsupported DATASET='${DATASET}'. Use DATASET=youtube or DATASET=ego4d."
  exit 1
fi

cd "${PROJECT_ROOT}"
source .venv/bin/activate

python - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available in current torch. Please install CUDA-enabled PyTorch in .venv.")
    print(f"torch version: {torch.__version__}")
    sys.exit(1)
try:
    torch.cuda.init()
    _ = torch.zeros(1, device="cuda")
except Exception as exc:
    print("ERROR: CUDA appears unavailable at runtime (often means running on login node).")
    print(f"Runtime CUDA check failed: {exc}")
    print("Please launch from a GPU allocation via srun/sbatch.")
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
  --model_name "baseline_sti_bert" \
  --task "STI" \
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
  "${WANDB_ARGS[@]}"
