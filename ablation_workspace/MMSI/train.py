import argparse
import random
from functools import partial
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from dataloader import SocialDataset, collate_fn
from model import MultimodalBaseline
from utils import AverageMeter, Progbar


seed = 1234
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class CenterLoss(torch.nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        self.centers = torch.nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, features, labels):
        centers_batch = self.centers.index_select(0, labels)
        return ((features - centers_batch) ** 2).sum(dim=1).mean() * 0.5


class SupConLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature,
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mask_sum = mask.sum(1)
        mask_sum = torch.clamp(mask_sum, min=1.0)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


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
    parser.add_argument('--language_model', type=str, default='bert', choices=['bert', 'roberta', 'electra'], help='Language model to use')
    parser.add_argument('--max_people_num', type=int, default=6, help='Maximum number of total players')
    parser.add_argument('--context_length', type=int, default=5, help='Size of conversation context')
    parser.add_argument('--batch_size', type=int, default=16, help='Mini-batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='Number of total epochs')
    parser.add_argument('--epochs_warmup', type=int, default=10, help='Number of visual warmup epochs')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--visual_film_layers', type=int, default=0, help='Number of FiLM layers in visual interaction encoder')
    parser.add_argument('--fusion_film_layers', type=int, default=0, help='Number of FiLM layers in aligned multimodal fusion module')
    parser.add_argument('--fusion_mode', type=str, default='standard', choices=['standard', 'bottleneck'],
                        help='Fusion module type for aligned multimodal fusion')
    parser.add_argument('--bottleneck_tokens', type=int, default=4,
                        help='Number of shared bottleneck tokens when fusion_mode=bottleneck')
    parser.add_argument('--bottleneck_fusion_layers', type=int, default=1,
                        help='Number of fusion layers using attention bottlenecks')
    parser.add_argument('--graph_layers', type=int, default=0,
                        help='Number of graph interaction layers over player nodes')
    parser.add_argument('--graph_mode', type=str, default='standard', choices=['standard', 'speaker_centered'],
                        help='Graph interaction mode for player relation modeling')
    parser.add_argument('--center_loss_weight', type=float, default=0.0, help='Weight for center loss, use 0.0 to disable')
    parser.add_argument('--center_loss_lr', type=float, default=0.5, help='Learning rate for center loss centers')
    parser.add_argument('--supcon_weight', type=float, default=0.0, help='Weight for supervised contrastive loss, use 0.0 to disable')
    parser.add_argument('--supcon_temperature', type=float, default=0.07, help='Temperature for supervised contrastive loss')
    parser.add_argument('--projection_dim', type=int, default=128, help='Projection head dimension for supervised contrastive loss')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='mmsi-baseline', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity/team')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')
    return parser.parse_args()

def get_tokenizer(language_model):
    if language_model == 'bert':
        from transformers import BertTokenizer
        return BertTokenizer.from_pretrained('bert-base-uncased')
    elif language_model == 'roberta':
        from transformers import RobertaTokenizer
        return RobertaTokenizer.from_pretrained('roberta-base')
    elif language_model == 'electra':
        from transformers import ElectraTokenizer
        return ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
    else:
        raise ValueError(f"Unsupported language model: {language_model}")

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch, args,
                    center_loss=None, center_optimizer=None, supcon_loss=None):
    model.train()
    train_loss = AverageMeter()
    ce_loss_meter = AverageMeter()
    center_loss_meter = AverageMeter()
    supcon_loss_meter = AverageMeter()
    progbar = Progbar(len(dataloader.dataset))

    for language_tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels in dataloader:
        optimizer.zero_grad()
        if center_optimizer is not None:
            center_optimizer.zero_grad()

        task_labels = task_labels.to(device)

        with torch.cuda.amp.autocast():
            outputs, features, projections_view1 = model(
                language_tokens, mask_idxs, keypoint_seqs, speaker_labels,
                warmup=(epoch < args.epochs_warmup), return_features=True
            )
            ce_loss = criterion(outputs, task_labels)
            center_term = outputs.new_tensor(0.0)
            supcon_term = outputs.new_tensor(0.0)
            if center_loss is not None:
                center_term = center_loss(features, task_labels)
            if supcon_loss is not None:
                _, _, projections_view2 = model(
                    language_tokens, mask_idxs, keypoint_seqs, speaker_labels,
                    warmup=(epoch < args.epochs_warmup), return_features=True
                )
                supcon_features = torch.stack([projections_view1, projections_view2], dim=1)
                supcon_term = supcon_loss(supcon_features, task_labels)
            loss = ce_loss + args.center_loss_weight * center_term + args.supcon_weight * supcon_term

        scaler.scale(loss).backward()
        if center_optimizer is not None:
            scaler.step(center_optimizer)
        scaler.step(optimizer)
        scaler.update()

        train_loss.update(loss.item(), task_labels.size(0))
        ce_loss_meter.update(ce_loss.item(), task_labels.size(0))
        center_loss_meter.update(center_term.item(), task_labels.size(0))
        supcon_loss_meter.update(supcon_term.item(), task_labels.size(0))
        progbar.add(args.batch_size, values=[('loss', loss.item())])

    return train_loss.avg, ce_loss_meter.avg, center_loss_meter.avg, supcon_loss_meter.avg

def evaluate(model, dataloader, device, epoch, args):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for language_tokens, mask_idxs, keypoint_seqs, speaker_labels, task_labels in dataloader:
            task_labels = task_labels.to(device)
            outputs = model(language_tokens, mask_idxs, keypoint_seqs, speaker_labels, warmup=(epoch < args.epochs_warmup))
            _, predicted = torch.max(outputs.data, 1)
            total += task_labels.size(0)
            correct += (predicted == task_labels).sum().item()

    return correct / total

def main():
    args = parse_args()
    if not 0.0 <= args.center_loss_weight <= 1.0:
        raise ValueError("--center_loss_weight must be in [0, 1].")
    if args.supcon_weight < 0.0:
        raise ValueError("--supcon_weight must be >= 0.")
    if args.bottleneck_tokens < 1:
        raise ValueError("--bottleneck_tokens must be >= 1.")
    if args.bottleneck_fusion_layers < 1:
        raise ValueError("--bottleneck_fusion_layers must be >= 1.")
    if args.graph_layers < 0:
        raise ValueError("--graph_layers must be >= 0.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_save_dir, exist_ok=True)

    model = MultimodalBaseline(
        args.max_people_num,
        args.language_model,
        visual_film_layers=args.visual_film_layers,
        fusion_film_layers=args.fusion_film_layers,
        projection_dim=args.projection_dim,
        fusion_mode=args.fusion_mode,
        bottleneck_tokens=args.bottleneck_tokens,
        bottleneck_fusion_layers=args.bottleneck_fusion_layers,
        graph_layers=args.graph_layers,
        graph_mode=args.graph_mode,
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

    language_params = [p for n, p in model.named_parameters() if 'convers_encoder' in n]
    other_params = [p for n, p in model.named_parameters() if 'convers_encoder' not in n]
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': args.learning_rate * 10},
        {'params': language_params, 'lr': args.learning_rate}
    ])
    center_loss = None
    center_optimizer = None
    supcon_loss = None
    if args.center_loss_weight > 0:
        center_loss = CenterLoss(args.max_people_num, 512).to(device)
        center_optimizer = torch.optim.SGD(center_loss.parameters(), lr=args.center_loss_lr)
    if args.supcon_weight > 0:
        supcon_loss = SupConLoss(temperature=args.supcon_temperature).to(device)

    tokenizer = get_tokenizer(args.language_model)
    args.tokenizer = tokenizer

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
        train_loss, ce_loss, center_term, supcon_term = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch, args, center_loss, center_optimizer, supcon_loss
        )
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
        print(f"Cross Entropy Loss: {ce_loss:.3f}")
        print(f"Center Loss: {center_term:.3f}")
        print(f"SupCon Loss: {supcon_term:.3f}")
        print(f"Test Accuracy: {test_acc:.3f}")
        print(f"Test Accuracy (Best): {best_acc:.3f} / {best_epoch + 1}e")
        print(f"Model: {args.model_name}")
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'cross_entropy_loss': ce_loss,
                'center_loss': center_term,
                'supcon_loss': supcon_term,
                'test_acc': test_acc,
                'best_test_acc': best_acc,
                'best_epoch': best_epoch + 1,
                'visual_film_layers': args.visual_film_layers,
                'fusion_film_layers': args.fusion_film_layers,
                'fusion_mode': args.fusion_mode,
                'bottleneck_tokens': args.bottleneck_tokens,
                'bottleneck_fusion_layers': args.bottleneck_fusion_layers,
                'graph_layers': args.graph_layers,
                'graph_mode': args.graph_mode,
                'center_loss_weight': args.center_loss_weight,
                'supcon_weight': args.supcon_weight,
            })

    if wandb_run is not None:
        wandb_run.finish()

if __name__ == '__main__':
    main()
