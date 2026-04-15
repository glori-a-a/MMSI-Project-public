import argparse
import os
import random
from functools import partial

import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from dataloader import SocialDataset, collate_fn
from model import MultimodalBaseline
from text_encoder import get_tokenizer
from utils import AverageMeter, Progbar


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
    parser.add_argument('--checkpoint_save_dir', type=str, default='./checkpoints', help='Directory for saving checkpoints')
    parser.add_argument('--video_dir', type=str, default=None, help='Directory containing extracted frames, videos, or cached arrays')
    parser.add_argument('--language_model', type=str, default='gpt2', choices=['bert', 'roberta', 'electra', 'gpt2', 'bart'], help='Language model to use')
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
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of total epochs')
    parser.add_argument('--epochs_warmup', type=int, default=10, help='Number of visual warmup epochs')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--max_train_samples', type=int, default=None, help='Optional cap on number of training samples')
    parser.add_argument('--max_test_samples', type=int, default=None, help='Optional cap on number of test samples')
    parser.add_argument('--sequence_length', type=int, default=16, help='Temporal sequence length for keypoints and frames')
    parser.add_argument('--video_fps', type=int, default=5, help='Frame rate used for aligned video sampling')
    parser.add_argument('--frame_size', type=int, default=224, help='Spatial size for extracted frames')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='mmsi-baseline', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity/team')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')
    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch, args):
    model.train()
    train_loss = AverageMeter()
    progbar = Progbar(len(dataloader.dataset))

    for language_tokens, token_positions, keypoint_seqs, speaker_labels, task_labels, visual_frames in dataloader:
        optimizer.zero_grad()
        language_tokens = language_tokens.to(device, non_blocking=True)
        token_positions = token_positions.to(device, non_blocking=True)
        keypoint_seqs = keypoint_seqs.to(device, non_blocking=True)
        speaker_labels = speaker_labels.to(device, non_blocking=True)
        task_labels = task_labels.to(device, non_blocking=True)
        if visual_frames is not None:
            visual_frames = visual_frames.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(
                language_tokens,
                token_positions,
                keypoint_seqs,
                speaker_labels,
                visual_frames=visual_frames,
                warmup=(epoch < args.epochs_warmup),
            )
            loss = criterion(outputs, task_labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss.update(loss.item(), task_labels.size(0))
        progbar.add(args.batch_size, values=[('loss', loss.item())])

    return train_loss.avg


def evaluate(model, dataloader, device, epoch, args):
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
                warmup=(epoch < args.epochs_warmup),
            )
            _, predicted = torch.max(outputs.data, 1)
            total += task_labels.size(0)
            correct += (predicted == task_labels).sum().item()

    return correct / total


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_save_dir, exist_ok=True)

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

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("W&B is enabled but wandb is not installed. Run: pip install wandb") from exc

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=vars(args),
        )

    language_params = [p for n, p in model.named_parameters() if 'text_encoder' in n or 'convers_encoder' in n]
    other_params = [p for n, p in model.named_parameters() if 'text_encoder' not in n and 'convers_encoder' not in n]
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': args.learning_rate * 10},
        {'params': language_params, 'lr': args.learning_rate}
    ])

    collate_fn_token = partial(collate_fn, tokenizer)
    train_dataset = SocialDataset(args, is_training=True)
    test_dataset = SocialDataset(args, is_training=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn_token,
                              shuffle=True, num_workers=args.workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn_token,
                             shuffle=False, num_workers=args.workers, drop_last=False)

    criterion = torch.nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_acc = 0
    best_epoch = 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, epoch, args)
        test_acc = evaluate(model, test_loader, device, epoch, args)

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save({
                'model_name': args.model_name,
                'model': model.state_dict(),
            }, f"{args.checkpoint_save_dir}/model.pt")

        print()
        print(f"Epoch: {epoch + 1}")
        print(f"Train Loss: {train_loss:.3f}")
        print(f"Test Accuracy: {test_acc:.3f}")
        print(f"Test Accuracy (Best): {best_acc:.3f} / {best_epoch + 1}e")
        print(f"Model: {args.model_name}")
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'test_acc': test_acc,
                'best_test_acc': best_acc,
                'best_epoch': best_epoch + 1,
                'language_model': args.language_model,
                'text_pooling': args.text_pooling,
                'visual_feature_type': args.visual_feature_type,
                'precomputed_visual_features': args.precomputed_visual_features,
                'fusion_strategy': args.fusion_strategy,
            })

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
