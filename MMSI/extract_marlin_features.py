import argparse
import os
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import SocialDataset, collate_fn
from marlin_wrapper import load_marlin
from text_encoder import get_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='STI', choices=['STI', 'PCR', 'MPP'])
    parser.add_argument('--txt_dir', type=str, required=True)
    parser.add_argument('--txt_labeled_dir', type=str, required=True)
    parser.add_argument('--keypoint_dir', type=str, required=True)
    parser.add_argument('--meta_dir', type=str, required=True)
    parser.add_argument('--data_split_file', type=str, required=True)
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--marlin_model_name', type=str, default='marlin_vit_base_ytf')
    parser.add_argument('--marlin_checkpoint', type=str, required=True)
    parser.add_argument('--language_model', type=str, default='roberta', choices=['bert', 'roberta', 'electra', 'gpt2', 'bart'])
    parser.add_argument('--context_length', type=int, default=5)
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--video_fps', type=int, default=5)
    parser.add_argument('--frame_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--max_train_samples', type=int, default=None)
    parser.add_argument('--max_test_samples', type=int, default=None)
    parser.add_argument('--split', type=str, default='both', choices=['train', 'test', 'both'])
    return parser.parse_args()


def encode_frames(marlin, frames, device):
    frames = frames.to(device, non_blocking=True).float() / 255.0
    frames = frames.permute(0, 4, 1, 2, 3)
    frames = F.interpolate(
        frames.permute(0, 2, 1, 3, 4).flatten(0, 1),
        size=(224, 224),
        mode='bilinear',
        align_corners=False,
    ).view(frames.size(0), frames.size(2), 3, 224, 224)
    frames = frames.permute(0, 2, 1, 3, 4)
    features = marlin.extract_features(frames, keep_seq=False)
    return features.cpu().numpy()


def cache_split(args, is_training, marlin, device):
    dataset = SocialDataset(args, is_training=is_training)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=partial(collate_fn, args.tokenizer),
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    saved = 0
    skipped = 0

    with torch.no_grad():
        offset = 0
        for _, _, _, _, _, frames in loader:
            batch_points = dataset.data_points[offset:offset + frames.size(0)]
            output_paths = [
                os.path.join(args.output_dir, f"{file_name}__{int(time_sec)}.npy")
                for _, _, _, file_name, time_sec, _, _, _ in batch_points
            ]
            missing = [not os.path.exists(path) for path in output_paths]
            if any(missing):
                features = encode_frames(marlin, frames, device)
                for feature, output_path, should_save in zip(features, output_paths, missing):
                    if should_save:
                        np.save(output_path, feature.reshape(1, -1).astype(np.float32))
                        saved += 1
                    else:
                        skipped += 1
            else:
                skipped += len(output_paths)
            offset += frames.size(0)
            print(f"cached={saved} skipped={skipped} processed={offset}/{len(dataset)}")


def main():
    args = parse_args()
    args.visual_feature_type = 'keypoint_marlin'
    args.tokenizer = get_tokenizer(args.language_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    marlin = load_marlin(args.marlin_model_name, checkpoint_path=args.marlin_checkpoint).to(device)
    marlin.eval()

    if args.split in ['train', 'both']:
        cache_split(args, is_training=True, marlin=marlin, device=device)
    if args.split in ['test', 'both']:
        cache_split(args, is_training=False, marlin=marlin, device=device)


if __name__ == '__main__':
    main()
