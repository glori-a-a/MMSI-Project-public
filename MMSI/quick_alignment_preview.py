import argparse
import json
import os
import re

import cv2
import numpy as np

from video_processor import build_temporal_indices


SKELETON_EDGES = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--txt_dir', required=True)
    parser.add_argument('--keypoint_dir', required=True)
    parser.add_argument('--data_split_file', required=True)
    parser.add_argument('--video_dir', required=True)
    parser.add_argument('--output_dir', default='./alignment_checks')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--video_fps', type=int, default=5)
    return parser.parse_args()


def get_time_in_seconds(utterance):
    time_str = utterance.split(': ')[0].split('(')[1][:-1]
    minute, second = map(int, time_str.split(':'))
    return minute * 60 + second


def get_video_size(video_dir, file_name, fallback_frame):
    for suffix in ('.mp4', '.avi', '.mov', '.mkv'):
        video_path = os.path.join(video_dir, f"{file_name}{suffix}")
        if os.path.exists(video_path):
            capture = cv2.VideoCapture(video_path)
            if capture.isOpened():
                width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                capture.release()
                if width > 0 and height > 0:
                    return float(width), float(height)
    frame_h, frame_w = fallback_frame.shape[:2]
    return float(frame_w), float(frame_h)


def draw_pose(frame, detections, original_width, original_height):
    overlay = frame.copy()
    frame_h, frame_w = frame.shape[:2]
    scale_x = frame_w / original_width
    scale_y = frame_h / original_height

    for detection in detections:
        points = np.asarray(detection['keypoints']).reshape(-1, 3)
        color = (
            int(50 + (detection['idx'] * 47) % 205),
            int(80 + (detection['idx'] * 73) % 175),
            int(120 + (detection['idx'] * 31) % 135),
        )

        for joint_a, joint_b in SKELETON_EDGES:
            point_a = points[joint_a]
            point_b = points[joint_b]
            if point_a[2] > 0.05 and point_b[2] > 0.05:
                ax = int(np.clip(point_a[0] * scale_x, 0, frame_w - 1))
                ay = int(np.clip(point_a[1] * scale_y, 0, frame_h - 1))
                bx = int(np.clip(point_b[0] * scale_x, 0, frame_w - 1))
                by = int(np.clip(point_b[1] * scale_y, 0, frame_h - 1))
                cv2.line(overlay, (ax, ay), (bx, by), color, 2)

        for point in points:
            if point[2] > 0.05:
                x = int(np.clip(point[0] * scale_x, 0, frame_w - 1))
                y = int(np.clip(point[1] * scale_y, 0, frame_h - 1))
                cv2.circle(overlay, (x, y), 3, color, -1)

    return overlay


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.data_split_file, 'r') as f:
        split = json.load(f)

    saved = 0
    for file_name in split['train'] + split['test']:
        if saved >= args.num_samples:
            break

        frame_dir = os.path.join(args.video_dir, f"{file_name}_frames_5fps")
        txt_file = os.path.join(args.txt_dir, f"{file_name}.txt")
        keypoint_file = os.path.join(args.keypoint_dir, f"{file_name}.npy")
        if not os.path.isdir(frame_dir) or not os.path.exists(txt_file) or not os.path.exists(keypoint_file):
            continue

        with open(txt_file, 'r') as f:
            utterances = [utterance for utterance in f.read().split('\n') if utterance]

        utterance = next((line for line in utterances if re.match(r'^\[Player\d+\]', line)), None)
        if utterance is None:
            continue

        time_sec = get_time_in_seconds(utterance)
        keypoint_data = np.load(keypoint_file, allow_pickle=True)
        frame_indices = build_temporal_indices(time_sec, args.sequence_length, args.video_fps, len(keypoint_data))

        panels = []
        original_width = original_height = None
        for frame_idx in frame_indices:
            frame_path = os.path.join(frame_dir, f"{frame_idx:06d}.png")
            frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if original_width is None:
                original_width, original_height = get_video_size(args.video_dir, file_name, frame)
            overlay = draw_pose(frame, keypoint_data[frame_idx], original_width, original_height)
            cv2.putText(
                overlay,
                f"{file_name} t={time_sec}s frame={frame_idx}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
            )
            panels.append(overlay)

        if len(panels) < 4:
            continue

        rows = []
        for start in range(0, min(len(panels), 16), 4):
            row_panels = panels[start:start + 4]
            if len(row_panels) < 4:
                break
            rows.append(np.concatenate(row_panels, axis=1))
        contact_sheet = np.concatenate(rows, axis=0)
        output_path = os.path.join(args.output_dir, f"alignment_{saved:02d}_{file_name}.png")
        cv2.imwrite(output_path, contact_sheet)
        print(f"Saved {output_path}")
        saved += 1

    if saved == 0:
        raise RuntimeError("No alignment previews were generated")


if __name__ == '__main__':
    main()
