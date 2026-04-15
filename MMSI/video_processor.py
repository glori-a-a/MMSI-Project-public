import json
import os
from pathlib import Path

import numpy as np


def build_temporal_indices(time_sec, sequence_length=16, fps=5, total_steps=None):
    start_step = fps * max(time_sec - 1, 0)
    indices = [start_step + offset for offset in range(sequence_length)]
    if total_steps is None:
        return indices
    return [min(max(idx, 0), total_steps - 1) for idx in indices]


def resize_frame(frame, frame_size):
    if frame.shape[0] == frame_size and frame.shape[1] == frame_size:
        return frame

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required to resize raw video frames") from exc

    return cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)


def extract_frames_from_video(video_path, output_dir, fps=5, frame_size=224, image_ext='.png'):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for frame extraction") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or fps
    frame_interval = max(int(round(source_fps / fps)), 1)

    frame_idx = 0
    saved_idx = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame_idx % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = resize_frame(frame, frame_size)
            output_path = output_dir / f"{saved_idx:06d}{image_ext}"
            cv2.imwrite(str(output_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            saved_idx += 1
        frame_idx += 1

    capture.release()
    return saved_idx


def _load_frame_folder(frame_dir):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required to read extracted frame folders") from exc

    frame_dir = Path(frame_dir)
    frame_paths = sorted(
        path for path in frame_dir.iterdir()
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    )
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in {frame_dir}")

    frames = []
    for frame_path in frame_paths:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not frames:
        raise RuntimeError(f"Failed to decode any frames from {frame_dir}")
    return np.stack(frames, axis=0)


def _load_frame_folder_sequence(frame_dir, indices, frame_size):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required to read extracted frame folders") from exc

    frame_dir = Path(frame_dir)
    frame_paths = sorted(
        path for path in frame_dir.iterdir()
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}
    )
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in {frame_dir}")

    last_valid = None
    frames = []
    for idx in indices:
        safe_idx = min(max(idx, 0), len(frame_paths) - 1)
        frame = cv2.imread(str(frame_paths[safe_idx]), cv2.IMREAD_COLOR)
        if frame is None:
            if last_valid is None:
                # Fall back to the nearest readable frame if the selected PNG is corrupt.
                for fallback_path in frame_paths:
                    fallback = cv2.imread(str(fallback_path), cv2.IMREAD_COLOR)
                    if fallback is not None:
                        frame = fallback
                        break
            else:
                frame = last_valid.copy()
        if frame is None:
            raise RuntimeError(f"Failed to decode frames from {frame_dir}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_frame(frame, frame_size)
        last_valid = frame
        frames.append(frame)

    return np.stack(frames, axis=0)


def get_video_frame_count(video_dir, file_name):
    video_dir = Path(video_dir)

    npy_path = video_dir / f"{file_name}.npy"
    if npy_path.exists():
        return len(np.load(npy_path, mmap_mode='r'))

    frame_dir = video_dir / file_name
    if frame_dir.is_dir():
        return len([
            path for path in frame_dir.iterdir()
            if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}
        ])

    extracted_frame_dir = video_dir / f"{file_name}_frames_5fps"
    if extracted_frame_dir.is_dir():
        return len([
            path for path in extracted_frame_dir.iterdir()
            if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}
        ])

    return None


def load_video_array(video_dir, file_name):
    video_dir = Path(video_dir)

    npy_path = video_dir / f"{file_name}.npy"
    if npy_path.exists():
        return np.load(npy_path)

    frame_dir = video_dir / file_name
    if frame_dir.is_dir():
        return _load_frame_folder(frame_dir)

    for suffix in ('.mp4', '.avi', '.mov', '.mkv'):
        video_path = video_dir / f"{file_name}{suffix}"
        if video_path.exists():
            temp_frame_dir = video_dir / f"{file_name}_frames_5fps"
            extract_frames_from_video(video_path, temp_frame_dir)
            return _load_frame_folder(temp_frame_dir)

    raise FileNotFoundError(f"Could not find video frames or video file for {file_name} in {video_dir}")


def load_frame_sequence(video_dir, file_name, time_sec, sequence_length=16, fps=5, frame_size=224):
    video_dir_path = Path(video_dir)

    for frame_dir in (video_dir_path / file_name, video_dir_path / f"{file_name}_frames_5fps"):
        if frame_dir.is_dir():
            total_steps = get_video_frame_count(video_dir, file_name)
            indices = build_temporal_indices(
                time_sec=time_sec,
                sequence_length=sequence_length,
                fps=fps,
                total_steps=total_steps,
            )
            return _load_frame_folder_sequence(frame_dir, indices, frame_size), indices

    video_array = load_video_array(video_dir, file_name)
    if video_array.ndim == 2:
        indices = build_temporal_indices(
            time_sec=time_sec,
            sequence_length=sequence_length,
            fps=fps,
            total_steps=len(video_array),
        )
        return video_array[indices], indices

    indices = build_temporal_indices(
        time_sec=time_sec,
        sequence_length=sequence_length,
        fps=fps,
        total_steps=len(video_array),
    )
    frames = [resize_frame(video_array[idx], frame_size) for idx in indices]
    return np.stack(frames, axis=0), indices


def save_alignment_map(alignment_map, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(alignment_map, f, indent=2)


def verify_alignment_pairs(keypoint_indices, frame_indices):
    if len(keypoint_indices) != len(frame_indices):
        return False
    return all(k_idx == f_idx for k_idx, f_idx in zip(keypoint_indices, frame_indices))
