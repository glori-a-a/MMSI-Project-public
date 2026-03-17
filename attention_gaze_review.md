# Gaze And Attention Bottlenecks Review

## Current MMSI Input Constraints

The current MMSI baseline in `ablation_workspace/MMSI` consumes:

- anonymized conversation text
- 2D keypoint tracks per player
- speaker identity labels

It does not currently consume:

- raw video frames
- face crops or head crops
- precomputed gaze vectors
- gaze confidence / uncertainty annotations

## Gaze360 Revisit

### Feasibility

Direct integration of `Gaze360` is not practical in the current training pipeline.

Reasons:

1. `Gaze360` is a gaze estimation model and dataset designed around visual inputs such as face or head crops rather than the 2D pose-only tensors used by this MMSI baseline.
2. The current dataloader only reads transcript files, metadata, and `.npy` pose/keypoint arrays. There is no frame loading path to run a gaze estimator online.
3. A proper integration would require an offline preprocessing pipeline:
   - decode original videos
   - track / crop each speaker and listener face or head region
   - run gaze estimation
   - align the resulting gaze vectors back to the 16-step social interaction windows
   - store gaze features for training

### Decision

Skip direct `Gaze360` integration for this milestone.

### If You Need A Follow-Up

The next realistic path is to build an offline gaze feature extractor and add:

- per-player 3D gaze direction
- gaze uncertainty score
- speaker-to-listener angular attention features

as extra visual channels alongside the existing pose representation.

## Attention Bottlenecks For Aligned Multimodal Fusion

### Feasibility

This is compatible with the current architecture and has now been implemented as an optional fusion path.

### What Was Added

In `ablation_workspace/MMSI/model.py`:

- `BottleneckFusionLayer`
- `BottleneckFusionStack`
- configurable fusion mode:
  - `standard`
  - `bottleneck`

In `ablation_workspace/MMSI/train.py`:

- `--fusion_mode`
- `--bottleneck_tokens`
- `--bottleneck_fusion_layers`

In `ablation_workspace/run_ablation_sti.sh`:

- `FUSION_MODE`
- `BOTTLENECK_TOKENS`
- `BOTTLENECK_FUSION_LAYERS`

In `ablation_workspace/submit_bottleneck_sweep.sh`:

- sweep script for bottleneck token counts and bottleneck fusion depth

### Implementation Mapping

The current MMSI fusion stage previously concatenated:

- `[CLS]`
- language token
- aligned visual tokens

and passed them through a standard transformer encoder.

The new bottleneck option instead:

1. keeps language tokens and visual tokens as separate streams
2. adds a small number of learned bottleneck tokens
3. allows cross-modal information exchange only through those bottleneck tokens

This keeps the change localized to the Aligned Multimodal Fusion module without changing the upstream encoders.

### Recommended First Sweep

- `VISUAL_FILM_LAYERS=3`
- `FUSION_FILM_LAYERS=1`
- `FUSION_MODE=bottleneck`
- `BOTTLENECK_TOKEN_OPTIONS="2 4 8"`
- `BOTTLENECK_LAYER_OPTIONS="1 2"`

## Additional Ideas

1. Replace the single masked language token representation with a short language token sequence before multimodal fusion.
2. Add speaker-listener graph edges and run graph attention over player nodes before fusion.
3. Add keypoint visibility / missingness as an explicit reliability feature.
4. Use a temporal consistency loss across adjacent windows for the same speaker.
5. Add pairwise speaker-to-listener relational tokens instead of only absolute positions.

## Sources

- Gaze360 GitHub: https://github.com/erkil1452/gaze360
- Attention Bottlenecks paper: https://proceedings.neurips.cc/paper/2021/file/76ba9f564ebbc35b1014ac498fafadd0-Paper.pdf
- Attention Bottlenecks code: https://github.com/NMS05/Multimodal-Fusion-with-Attention-Bottlenecks
