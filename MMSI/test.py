import argparse
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import SocialDataset, collate_fn
from model import MultimodalBaseline
from text_encoder import get_tokenizer


seed = 1234
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='model_name', help='Name of the model')
    parser.add_argument('--task', type=str, default='STI', choices=['STI', 'PCR', 'MPP'], help='Task to perform')
    parser.add_argument('--txt_dir', type=str, default='enter_the_path', help='Directory of anonymized transcripts')
    parser.add_argument('--txt_labeled_dir', type=str, default='enter_the_path', help='Directory of labeled anonymized transcripts')
    parser.add_argument('--keypoint_dir', type=str, default='enter_the_path', help='Directory of keypoints')
    parser.add_argument('--meta_dir', type=str, default='enter_the_path', help='Directory of game meta data')
    parser.add_argument('--data_split_file', type=str, default='enter_the_path', help='File path for data split')
    parser.add_argument('--checkpoint_file', type=str, default='enter_the_path', help='File path for loading checkpoint')
    parser.add_argument('--video_dir', type=str, default=None, help='Directory containing extracted frames, videos, or cached arrays')
    parser.add_argument('--language_model', type=str, default='roberta', choices=['bert', 'roberta', 'electra', 'gpt2', 'bart'], help='Language model to use')
    parser.add_argument('--text_pooling', type=str, default='auto', choices=['auto', 'mask', 'last', 'mean'], help='Text pooling strategy')
    parser.add_argument('--visual_feature_type', type=str, default='keypoint',
                        choices=['keypoint', 'vit', 'keypoint_vit', 'marlin', 'keypoint_marlin'],
                        help='Visual representation to use')
    parser.add_argument('--marlin_model_name', type=str, default='marlin_vit_base_ytf', help='MARLIN backbone variant')
    parser.add_argument('--marlin_checkpoint', type=str, default=None, help='Local MARLIN encoder/full checkpoint path')
    parser.add_argument('--marlin_from_online', action='store_true', help='Download MARLIN weights from upstream at runtime')
    parser.add_argument('--precomputed_visual_features', action='store_true',
                        help='Treat --video_dir as cached 768-d visual features instead of raw frames')
    parser.add_argument('--fusion_strategy', type=str, default='gated', choices=['gated', 'late_concat'],
                        help='Strategy for combining keypoints with video features')
    parser.add_argument('--max_people_num', type=int, default=6, help='Maximum number of total players')
    parser.add_argument('--context_length', type=int, default=5, help='Size of conversation context')
    parser.add_argument('--batch_size', type=int, default=16, help='Mini-batch size')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--max_test_samples', type=int, default=None, help='Optional cap on number of test samples')
    parser.add_argument('--sequence_length', type=int, default=16, help='Temporal sequence length for keypoints and frames')
    parser.add_argument('--video_fps', type=int, default=5, help='Frame rate used for aligned video sampling')
    parser.add_argument('--frame_size', type=int, default=224, help='Spatial size for extracted frames')
    return parser.parse_args()


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for language_tokens, token_positions, keypoint_seqs, speaker_labels, task_labels, visual_frames in dataloader:
            language_tokens = language_tokens.to(device, non_blocking=True)
            token_positions = token_positions.to(device, non_blocking=True)
            keypoint_seqs = keypoint_seqs.to(device, non_blocking=True)
            speaker_labels = speaker_labels.to(device, non_blocking=True)
            task_labels = task_labels.to(device, non_blocking=True)
            if visual_frames is not None:
                visual_frames = visual_frames.to(device, non_blocking=True)
            outputs = model(
                language_tokens,
                token_positions,
                keypoint_seqs,
                speaker_labels,
                visual_frames=visual_frames,
                warmup=False,
            )
            _, predicted = torch.max(outputs.data, 1)
            total += task_labels.size(0)
            correct += (predicted == task_labels).sum().item()

    return correct / total


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(args.language_model)
    args.tokenizer = tokenizer

    model = MultimodalBaseline(
        args.max_people_num,
        args.language_model,
        tokenizer=tokenizer,
        text_pooling=args.text_pooling,
        visual_feature_type=args.visual_feature_type,
        marlin_model_name=args.marlin_model_name,
        marlin_checkpoint=args.marlin_checkpoint,
        marlin_from_online=args.marlin_from_online,
        precomputed_visual_features=args.precomputed_visual_features,
        fusion_strategy=args.fusion_strategy,
    ).to(device)

    checkpoint = torch.load(args.checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model'])
    print(f"Loaded checkpoint from {args.checkpoint_file}")

    collate_fn_token = partial(collate_fn, tokenizer)
    test_dataset = SocialDataset(args, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn_token,
                             shuffle=False, num_workers=args.workers, drop_last=False)

    test_acc = evaluate(model, test_loader, device)

    print(f"Test Accuracy: {test_acc:.3f}")
    print(f"Model: {args.model_name}")


if __name__ == '__main__':
    main()
