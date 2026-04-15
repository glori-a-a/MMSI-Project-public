import argparse
import os
import random

import cv2
import numpy as np

from dataloader import SocialDataset
from text_encoder import get_tokenizer
from video_processor import load_frame_sequence


SKELETON_EDGES = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='STI', choices=['STI', 'PCR', 'MPP'])
    parser.add_argument('--txt_dir', type=str, required=True)
    parser.add_argument('--txt_labeled_dir', type=str, required=True)
    parser.add_argument('--keypoint_dir', type=str, required=True)
    parser.add_argument('--meta_dir', type=str, required=True)
    parser.add_argument('--data_split_file', type=str, required=True)
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--language_model', type=str, default='roberta', choices=['bert', 'roberta', 'electra', 'gpt2', 'bart'])
    parser.add_argument('--visual_feature_type', type=str, default='keypoint_vit')
    parser.add_argument('--context_length', type=int, default=5)
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--video_fps', type=int, default=5)
    parser.add_argument('--frame_size', type=int, default=224)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='./alignment_checks')
    parser.add_argument('--require_existing_frames', action='store_true',
                        help='Only sample files that already have *_frames_5fps folders')
    return parser.parse_args()


def draw_keypoints(frame, keypoints):
    overlay = frame.copy()
    points = keypoints.reshape(-1, 2)
    frame_h, frame_w = frame.shape[:2]

    for joint_a, joint_b in SKELETON_EDGES:
        point_a = points[joint_a]
        point_b = points[joint_b]
        if np.any(point_a) and np.any(point_b):
            ax = int(np.clip((point_a[0] + 1.0) * 0.5 * frame_w, 0, frame_w - 1))
            ay = int(np.clip((point_a[1] + 1.0) * 0.5 * frame_h, 0, frame_h - 1))
            bx = int(np.clip((point_b[0] + 1.0) * 0.5 * frame_w, 0, frame_w - 1))
            by = int(np.clip((point_b[1] + 1.0) * 0.5 * frame_h, 0, frame_h - 1))
            cv2.line(overlay, (ax, ay), (bx, by), (0, 255, 0), 2)

    for point in points:
        if np.any(point):
            x = int(np.clip((point[0] + 1.0) * 0.5 * frame_w, 0, frame_w - 1))
            y = int(np.clip((point[1] + 1.0) * 0.5 * frame_h, 0, frame_h - 1))
            cv2.circle(overlay, (x, y), 3, (255, 0, 0), -1)

    return overlay


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = get_tokenizer(args.language_model)
    args.tokenizer = tokenizer

    dataset = SocialDataset(args, is_training=False)
    candidate_indices = list(range(len(dataset)))
    if args.require_existing_frames:
        candidate_indices = [
            idx for idx in candidate_indices
            if os.path.isdir(os.path.join(args.video_dir, f"{dataset.data_points[idx][3]}_frames_5fps"))
        ]
        if not candidate_indices:
            raise RuntimeError(f"No dataset samples have existing *_frames_5fps folders in {args.video_dir}")

    sample_indices = random.sample(candidate_indices, min(args.num_samples, len(candidate_indices)))

    for output_i, sample_idx in enumerate(sample_indices):
        _, _, keypoint_seq, file_name, time_sec, _, speaker_label, _ = dataset.data_points[sample_idx]
        frames, frame_indices = load_frame_sequence(
            args.video_dir,
            file_name=file_name,
            time_sec=time_sec,
            sequence_length=args.sequence_length,
            fps=args.video_fps,
            frame_size=args.frame_size,
        )

        clip_frames = []
        for frame_i, frame in enumerate(frames):
            frame_bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            overlay = draw_keypoints(frame_bgr, keypoint_seq[speaker_label, frame_i])
            caption = f"{file_name} t={time_sec}s idx={frame_indices[frame_i]}"
            cv2.putText(overlay, caption, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            clip_frames.append(overlay)

        contact_sheet = np.concatenate(clip_frames[:4], axis=1)
        for extra_row_start in range(4, len(clip_frames), 4):
            row = np.concatenate(clip_frames[extra_row_start:extra_row_start + 4], axis=1)
            if row.shape[1] != contact_sheet.shape[1]:
                pad_width = contact_sheet.shape[1] - row.shape[1]
                row = cv2.copyMakeBorder(row, 0, 0, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            contact_sheet = np.concatenate([contact_sheet, row], axis=0)

        output_path = os.path.join(args.output_dir, f"alignment_{output_i:02d}_{file_name}.png")
        cv2.imwrite(output_path, contact_sheet)
        print(f"Saved alignment visualization to {output_path}")


if __name__ == '__main__':
    main()
