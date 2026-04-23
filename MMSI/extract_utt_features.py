#!/usr/bin/env python3
"""
extract_utt_features.py
=======================

Per-utterance visual feature extraction for the MMSI "Werewolf Among Us"
YouTube split (same convention used in the CVPR 2024 MMSI paper).

Motivation
----------
The original pipeline first dumps every frame of every video to disk at some
sampling rate, then the dataloader reads per-utterance windows out of that dump.
That is wasteful in both I/O and storage. This script skips the intermediate
frame-dump step: it reads each transcript, and for every utterance it
   (1) seeks directly into the raw mp4 at the utterance timestamp,
   (2) samples a short window of frames around that timestamp,
   (3) runs ViT or MARLIN over those frames,
   (4) writes a time-structured (T, D) feature array per utterance, keyed by
       `{video_stem}__{time_sec}.npy` — matching the existing cache convention
       used by `extract_vit_features.py` / `extract_marlin_features.py`.

Transcript format (MMSI benchmark, anonymized):
    [Player1] (00:02): text ... [optional label tags]
    [Player3] (00:13): text ...

Usage
-----
    python extract_utt_features.py \
        --video_dir      data/raw_video_sources/werewolf_among_us/Youtube/videos_mmsi_named \
        --transcript_dir data/mmsi/benchmark/youtube/transcripts/anonymized \
        --out_dir        data/mmsi/cache/youtube_vit_per_utterance \
        --encoder        vit \
        --sequence_length 16 \
        --video_fps       5 \
        --device          cuda
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------
# Matches lines like:  "[Player3] (02:15): I think Player1 is the werewolf. [labelA]"
UTTERANCE_RE = re.compile(
    r"""
    ^\s*
    \[Player(?P<speaker>\d+)\]          # speaker token, e.g. [Player3]
    \s*
    \((?P<mm>\d{1,2}):(?P<ss>\d{2})\)   # timestamp, e.g. (02:15)
    \s*:\s*
    (?P<text>.*?)                       # utterance text (non-greedy)
    \s*$
    """,
    re.VERBOSE,
)


def parse_transcript(transcript_path: Path) -> List[dict]:
    """Parse one MMSI transcript file into a list of utterances."""
    utterances: List[dict] = []
    with transcript_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = UTTERANCE_RE.match(line)
            if not m:
                continue
            time_sec = int(m.group("mm")) * 60 + int(m.group("ss"))
            raw_text = m.group("text")
            # Strip trailing [label] tags but keep the sentence itself.
            text_no_labels = re.sub(r"\[[^\]]+\]", "", raw_text).strip()
            speaker_idx = int(m.group("speaker"))
            utterances.append(
                {
                    "speaker": f"Player{speaker_idx}",
                    "speaker_idx": speaker_idx,
                    "time_sec": time_sec,
                    "text": text_no_labels,
                    "raw_line": line,
                }
            )
    return utterances


# ---------------------------------------------------------------------------
# Per-utterance frame sampling (direct seek, no intermediate frame dump)
# ---------------------------------------------------------------------------
def sample_frames_for_utterance(
    cap: cv2.VideoCapture,
    center_sec: float,
    sequence_length: int,
    video_fps: int,
    frame_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Seek into the already-opened `cap` and uniformly sample `sequence_length`
    frames across a window of (sequence_length / video_fps) seconds, centered
    on `center_sec`.

    Returns: (T, H, W, 3) uint8 BGR.
    """
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_sec = max(0.0, total_frames / native_fps) if total_frames > 0 else 0.0

    window_sec = sequence_length / float(video_fps)
    if duration_sec <= 0.0:
        start = max(0.0, center_sec - window_sec / 2.0)
        end = start + window_sec
    else:
        clamped_center = min(max(center_sec, 0.0), duration_sec)
        start = max(0.0, clamped_center - window_sec / 2.0)
        end = min(duration_sec, clamped_center + window_sec / 2.0)
        if end <= start:
            end = min(duration_sec, start + max(window_sec, 1e-3))
        start = max(0.0, end - window_sec)  # re-clamp if we hit the tail

    if sequence_length == 1:
        target_times = [0.5 * (start + end)]
    else:
        target_times = np.linspace(start, end, num=sequence_length, endpoint=True, dtype=np.float64)

    frames = np.zeros((sequence_length, frame_size[1], frame_size[0], 3), dtype=np.uint8)
    last_good: Optional[np.ndarray] = None
    for i, t in enumerate(target_times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            if last_good is not None:
                frames[i] = last_good
            continue
        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
        frames[i] = frame
        last_good = frame

    # Back-fill any leading black frames with the first good frame we found.
    if last_good is not None:
        for i in range(sequence_length):
            if frames[i].sum() == 0:
                frames[i] = last_good
    else:
        raise RuntimeError(f"Failed to decode any frames near t={center_sec:.3f}s")
    return frames


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
def load_encoder(
    encoder_name: str,
    device: str,
    sequence_length: int,
    frame_size: Tuple[int, int],
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function mapping (T, H, W, 3) uint8 BGR frames -> (T, D) np.float32."""
    if encoder_name == "vit":
        from transformers import ViTImageProcessor, ViTModel

        model_name = "google/vit-base-patch16-224-in21k"
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTModel.from_pretrained(model_name).to(device).eval()

        @torch.no_grad()
        def encode(frames_bgr: np.ndarray) -> np.ndarray:
            frames_rgb = frames_bgr[..., ::-1]  # BGR -> RGB
            inputs = processor(images=list(frames_rgb), return_tensors="pt").to(device)
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :]      # (T, D)
            return cls.cpu().numpy()                  # (T, D)

        return encode

    if encoder_name == "marlin":
        from marlin_wrapper import load_marlin

        model = load_marlin(
            model_name="marlin_vit_base_ytf",
            checkpoint_path=getattr(load_encoder, "_marlin_checkpoint", None),
            from_online=getattr(load_encoder, "_marlin_from_online", False),
        ).to(device).eval()

        @torch.no_grad()
        def encode(frames_bgr: np.ndarray) -> np.ndarray:
            frames_rgb = frames_bgr[..., ::-1].copy()
            clip = torch.from_numpy(frames_rgb).float() / 255.0  # (T, H, W, 3)
            clip = clip.permute(3, 0, 1, 2).unsqueeze(0).to(device)  # (1, C, T, H, W)
            feat = model.extract_features(clip, keep_seq=True)        # (1, N, D)

            tubelet_size = getattr(model, "tubelet_size", 2)
            temporal_steps = max(sequence_length // tubelet_size, 1)
            spatial_steps = (frame_size[0] // 16) * (frame_size[1] // 16)
            feat = feat.view(1, temporal_steps, spatial_steps, -1).mean(dim=2)  # (1, T', D)

            # Expand tubelet tokens back to frame-aligned time steps so the
            # downstream MMSI fusion stack sees the same temporal granularity
            # as the ViT cache path.
            feat = feat.repeat_interleave(tubelet_size, dim=1)
            feat = feat[:, :sequence_length, :]
            if feat.size(1) < sequence_length:
                pad = feat[:, -1:, :].repeat(1, sequence_length - feat.size(1), 1)
                feat = torch.cat([feat, pad], dim=1)
            return feat.squeeze(0).cpu().numpy()                     # (T, D)

        return encode

    raise ValueError(f"Unknown encoder: {encoder_name!r}")


def resolve_video_path(video_dir: Path, stem: str, explicit_ext: Optional[str]) -> Optional[Path]:
    if explicit_ext:
        candidate = video_dir / f"{stem}{explicit_ext}"
        return candidate if candidate.exists() else None

    for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        candidate = video_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Per-video driver
# ---------------------------------------------------------------------------
def process_video(
    video_path: Path,
    transcript_path: Path,
    out_dir: Path,
    encoder_fn: Callable[[np.ndarray], np.ndarray],
    sequence_length: int,
    video_fps: int,
    frame_size: Tuple[int, int],
    save_raw_frames: bool,
) -> None:
    utterances = parse_transcript(transcript_path)
    if not utterances:
        print(f"[WARN] No utterances parsed from {transcript_path.name}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video {video_path}")
        return

    video_stem = video_path.stem
    video_out = out_dir / video_stem
    video_out.mkdir(parents=True, exist_ok=True)

    manifest: List[dict] = []
    for utt in tqdm(utterances, desc=video_stem, leave=False):
        t = utt["time_sec"]
        feat_path = video_out / f"{video_stem}__{t}.npy"
        if feat_path.exists():
            feat = np.load(feat_path)
        else:
            frames = sample_frames_for_utterance(
                cap,
                center_sec=t,
                sequence_length=sequence_length,
                video_fps=video_fps,
                frame_size=frame_size,
            )
            feat = encoder_fn(frames).astype(np.float32)
            np.save(feat_path, feat)

            if save_raw_frames:
                np.save(video_out / f"{video_stem}__{t}__frames.npy", frames)

        manifest.append(
            {
                "speaker": utt["speaker"],
                "time_sec": t,
                "text": utt["text"],
                "feature_path": str(feat_path.relative_to(out_dir)),
            }
        )

    cap.release()
    (video_out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video_dir", required=True, type=Path)
    ap.add_argument("--transcript_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--encoder", choices=["vit", "marlin"], default="vit")
    ap.add_argument("--sequence_length", type=int, default=16,
                    help="Frames sampled per utterance. Matches the default in your existing dataloader.")
    ap.add_argument("--video_fps", type=int, default=5,
                    help="Effective sampling fps. Together with --sequence_length this fixes the utterance window in seconds.")
    ap.add_argument("--frame_size", type=int, nargs=2, default=[224, 224],
                    help="H W after resize. Default 224 224 for ViT / MARLIN.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_raw_frames", action="store_true",
                    help="Also cache the raw (T,H,W,3) frame arrays so a different encoder can be re-run later without re-reading videos.")
    ap.add_argument("--video_ext", default=None,
                    help="Optional fixed extension, e.g. .mp4. If omitted, common video extensions are tried automatically.")
    ap.add_argument("--marlin_checkpoint", type=Path, default=None,
                    help="Optional local MARLIN checkpoint. Required unless --marlin_from_online is set.")
    ap.add_argument("--marlin_from_online", action="store_true",
                    help="Allow MARLIN weights to be fetched online instead of loading a local checkpoint.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    load_encoder._marlin_checkpoint = str(args.marlin_checkpoint) if args.marlin_checkpoint else None
    load_encoder._marlin_from_online = args.marlin_from_online
    encoder_fn = load_encoder(
        args.encoder,
        args.device,
        sequence_length=args.sequence_length,
        frame_size=tuple(args.frame_size),
    )

    transcripts = sorted(args.transcript_dir.glob("*.txt"))
    print(f"Found {len(transcripts)} transcripts in {args.transcript_dir}")
    for tpath in tqdm(transcripts, desc="videos"):
        vpath = resolve_video_path(args.video_dir, tpath.stem, args.video_ext)
        if vpath is None:
            print(f"[SKIP] Missing video for transcript {tpath.name}")
            continue
        process_video(
            video_path=vpath,
            transcript_path=tpath,
            out_dir=args.out_dir,
            encoder_fn=encoder_fn,
            sequence_length=args.sequence_length,
            video_fps=args.video_fps,
            frame_size=tuple(args.frame_size),
            save_raw_frames=args.save_raw_frames,
        )


if __name__ == "__main__":
    main()
