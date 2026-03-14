# MMSI Project

This repository contains:

- the original `MMSI` baseline code
- local setup and training scripts for running the baseline
- an `ablation_workspace` with the architecture extensions and experiment scripts
- a slide draft summarizing the architecture and experimental results

## Structure

- `MMSI/`: original baseline codebase used for reproduction
- `ablation_workspace/`: modified codebase for FiLM, Center Loss, and SupCon experiments
- `run_baseline_sti.sh`: baseline training entrypoint
- `download_mmsi_data.sh`: dataset download helper based on the original README links
- `slides_mmsi_ablation.md`: slide draft for architecture and results

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
