# MMSI Project

This repository contains:

- the original `MMSI` baseline code
- local setup and training scripts for running the baseline
- an `ablation_workspace` with the architecture extensions and experiment scripts
- a `fixed1.0` branch with the current text/video encoder work

## Structure

- `MMSI/`: original baseline codebase used for reproduction
- `ablation_workspace/`: modified codebase for FiLM, Center Loss, and SupCon experiments
- `run_baseline_sti.sh`: baseline training entrypoint
- `download_mmsi_data.sh`: dataset download helper based on the original README links

## `fixed1.0` branch

This branch contains the current MMSI text/video feature exploration work:

- GPT-2, BART, RoBERTa, BERT, and ELECTRA text encoder support in `MMSI/text_encoder.py`
- decoder-only GPT-2 pooling and encoder-side BART feature extraction
- keypoint-video temporal alignment utilities at 5fps
- ViT frame feature extraction and cached ViT training scripts
- MARLIN wrapper code for local integration with `third_party/MARLIN`

Useful entrypoints:

- `run_gpt2_sti_wandb.sh`
- `run_bart_sti_wandb.sh`
- `run_gpt2_sti_keypoint_vit_quick_wandb.sh`
- `run_extract_vit_cache_youtube_quick.sh`
- `run_gpt2_sti_cached_vit_quick_wandb.sh`
- `MMSI/quick_alignment_preview.py`

## Notes

- Large runtime artifacts are intentionally excluded from version control:
  - datasets
  - checkpoints
  - W&B logs
  - virtual environments

- Dataset download links are documented in:
  - `MMSI/README.md`

## Best Completed Result

On the YouTube STI setup, the best completed configuration in `ablation_workspace` is:

- `visual_film_layers = 3`
- `fusion_film_layers = 1`
- `center_loss_weight = 0.0`

Best test accuracy:

- `0.72519`
