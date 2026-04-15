import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from marlin_wrapper import load_marlin
from text_encoder import TextEncoderFactory


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('encoding', encoding, persistent=False)

    def forward(self, x):
        batch_size = x.size(1)
        pos_enc = self.encoding[:x.size(0)].detach().unsqueeze(1)
        pos_enc = torch.tile(pos_enc, (1, batch_size, 1))
        return x + pos_enc


class MultimodalBaseline(nn.Module):
    def __init__(self, class_num, language_model, tokenizer, text_pooling='auto', visual_feature_type='keypoint',
                 marlin_model_name='marlin_vit_base_ytf', marlin_checkpoint=None, marlin_from_online=False,
                 precomputed_visual_features=False):
        super(MultimodalBaseline, self).__init__()

        self.class_num = class_num
        self.language_model = language_model
        self.visual_feature_type = visual_feature_type
        self.uses_visual_frames = visual_feature_type in ['vit', 'keypoint_vit', 'marlin', 'keypoint_marlin']
        self.uses_keypoints = visual_feature_type in ['keypoint', 'keypoint_vit', 'keypoint_marlin']
        self.use_visual_only = visual_feature_type in ['vit', 'marlin']
        self.marlin_model_name = marlin_model_name
        self.precomputed_visual_features = precomputed_visual_features
        self.visual_fusion_logit = nn.Parameter(torch.tensor(-2.0))

        self.text_encoder = TextEncoderFactory(language_model, tokenizer, pooling=text_pooling)
        self.convers_encoder = self.text_encoder.encoder

        self.convers_fc = nn.Sequential(
            nn.Linear(self.text_encoder.hidden_size, 512))

        self.coordinate_fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 64))

        self.speaker_encoder = nn.Sequential(
            nn.Linear(9 * 64, 512),
            Permute(*(0, 2, 1)),
            nn.BatchNorm1d(512),
            Permute(*(0, 2, 1)),
            nn.ReLU(),
            nn.Linear(512, 512),
            Permute(*(0, 2, 1)),
            nn.BatchNorm1d(512),
            Permute(*(0, 2, 1)),
            nn.ReLU(),
            nn.Linear(512, 512),
            Permute(*(0, 2, 1)),
            nn.BatchNorm1d(512),
            Permute(*(0, 2, 1)),
            nn.ReLU(),
            nn.Linear(512, 512))

        self.position_encoder = nn.Sequential(
            nn.Linear(6 * 64, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512))

        self.position_fc = nn.Sequential(
            nn.Linear(512, 512))

        self.onehot_encoder = nn.Sequential(
            nn.Linear(class_num, 512))

        visual_trans_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024)
        self.visual_trans = nn.TransformerEncoder(visual_trans_layer, num_layers=3)

        multi_trans_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=1024)
        self.multi_trans = nn.TransformerEncoder(multi_trans_layer, num_layers=2)
        self.multi_trans_pre = nn.TransformerEncoder(multi_trans_layer, num_layers=2)

        self.positional_enc = PositionalEncoding(d_model=512, max_len=20)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))

        self.vit_encoder = None
        self.visual_fc = None
        self.marlin_encoder = None

        if self.uses_visual_frames:
            if precomputed_visual_features:
                self.visual_fc = nn.Linear(768, 512)
            if visual_feature_type in ['vit', 'keypoint_vit']:
                if not precomputed_visual_features:
                    from transformers import ViTModel
                    self.vit_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
                    self.visual_fc = nn.Linear(self.vit_encoder.config.hidden_size, 512)
            elif visual_feature_type in ['marlin', 'keypoint_marlin']:
                if not precomputed_visual_features:
                    self.marlin_encoder = load_marlin(
                        model_name=marlin_model_name,
                        checkpoint_path=marlin_checkpoint,
                        from_online=marlin_from_online,
                    )
                    marlin_hidden_size = self.marlin_encoder.encoder.embed_dim
                    self.visual_fc = nn.Linear(marlin_hidden_size, 512)

        self.classifier = nn.Sequential(
            nn.Linear(512, class_num))

        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        self.register_buffer('imagenet_mean', imagenet_mean, persistent=False)
        self.register_buffer('imagenet_std', imagenet_std, persistent=False)

    def encode_text(self, language_token, token_positions):
        attention_mask = (language_token != self.text_encoder.pad_token_id).float()
        convers_feature = self.text_encoder(language_token, attention_mask, token_positions)
        return self.convers_fc(convers_feature).unsqueeze(0)

    def encode_keypoints(self, keypoint_seqs, speaker_labels):
        batch_size = speaker_labels.size(0)
        seq_len = keypoint_seqs.size(2)
        gaze_feature, gesture_feature = [], []
        for batch_i in range(batch_size):
            gaze_feature.append(keypoint_seqs[batch_i:batch_i + 1, speaker_labels[batch_i], :, 0:3 * 2])
            gesture_feature.append(keypoint_seqs[batch_i:batch_i + 1, speaker_labels[batch_i], :, 5 * 2:11 * 2])

        speaker_feature = torch.concat([torch.concat(gaze_feature, dim=0), torch.concat(gesture_feature, dim=0)], dim=-1)
        speaker_feature = speaker_feature.view(batch_size, seq_len, -1, 2)
        speaker_feature = self.coordinate_fc(speaker_feature).view(batch_size, seq_len, -1)
        speaker_feature = self.speaker_encoder(speaker_feature).permute(1, 0, 2)

        speaker_onehot = torch.nn.functional.one_hot(speaker_labels, num_classes=self.class_num).float()
        speaker_onehot_feature = self.onehot_encoder(speaker_onehot)

        position_feature = keypoint_seqs[:, :, 5, 0:2]
        position_feature = self.coordinate_fc(position_feature).view(batch_size, -1)
        position_feature = self.position_encoder(position_feature) + speaker_onehot_feature
        position_feature = self.position_fc(position_feature).unsqueeze(0)

        vis_feature = torch.concat([position_feature, self.positional_enc(speaker_feature[::2, :, :])], 0)
        vis_feature = self.visual_trans(vis_feature)
        return self.positional_enc(vis_feature)

    def encode_visual_frames(self, visual_frames):
        if visual_frames is None:
            raise ValueError(f"visual_feature_type={self.visual_feature_type} requires visual inputs")

        if visual_frames.dim() == 3:
            frame_features = self.visual_fc(visual_frames)
            if self.uses_keypoints:
                global_token = frame_features.mean(dim=1, keepdim=True)
                frame_features = torch.cat([global_token, frame_features[:, ::2, :]], dim=1)
            return frame_features.permute(1, 0, 2)

        if self.visual_feature_type in ['marlin', 'keypoint_marlin']:
            if self.marlin_encoder is None:
                raise ValueError("MARLIN encoder is not initialized")
            frames = visual_frames.float() / 255.0
            frames = frames.permute(0, 4, 1, 2, 3)
            frames = F.interpolate(
                frames.permute(0, 2, 1, 3, 4).flatten(0, 1),
                size=(224, 224),
                mode='bilinear',
                align_corners=False,
            ).view(visual_frames.size(0), visual_frames.size(1), 3, 224, 224)
            frames = frames.permute(0, 2, 1, 3, 4)
            marlin_features = self.marlin_encoder.extract_features(frames, keep_seq=False)
            frame_features = self.visual_fc(marlin_features).unsqueeze(1)
            if self.uses_keypoints:
                frame_features = frame_features.repeat(1, 9, 1)
            return frame_features.permute(1, 0, 2)

        if self.vit_encoder is None:
            raise ValueError(f"Raw frame encoding is not configured for {self.visual_feature_type}")

        frames = visual_frames.float() / 255.0
        frames = frames.permute(0, 1, 4, 2, 3)
        frames = F.interpolate(
            frames.flatten(0, 1),
            size=(224, 224),
            mode='bilinear',
            align_corners=False,
        ).view(visual_frames.size(0), visual_frames.size(1), 3, 224, 224)
        frames = (frames - self.imagenet_mean) / self.imagenet_std

        batch_size, seq_len = frames.size(0), frames.size(1)
        vit_outputs = self.vit_encoder(pixel_values=frames.flatten(0, 1)).last_hidden_state[:, 0]
        vit_outputs = self.visual_fc(vit_outputs.view(batch_size, seq_len, -1))
        if self.uses_keypoints:
            global_token = vit_outputs.mean(dim=1, keepdim=True)
            vit_outputs = torch.cat([global_token, vit_outputs[:, ::2, :]], dim=1)
        return vit_outputs.permute(1, 0, 2)

    def forward(self, language_token, token_positions, keypoint_seqs, speaker_labels, visual_frames=None, warmup=False):
        convers_feature = self.encode_text(language_token, token_positions)

        visual_streams = []
        if self.uses_keypoints:
            visual_streams.append(self.encode_keypoints(keypoint_seqs, speaker_labels))
        if self.uses_visual_frames:
            visual_streams.append(self.positional_enc(self.encode_visual_frames(visual_frames)))

        if not visual_streams:
            raise ValueError("At least one visual stream must be enabled")

        if len(visual_streams) == 1:
            vis_feature = visual_streams[0]
        else:
            visual_weight = torch.sigmoid(self.visual_fusion_logit)
            vis_feature = (1.0 - visual_weight) * visual_streams[0] + visual_weight * visual_streams[1]

        batch_size = speaker_labels.size(0)
        cls_tokens = self.cls_token.repeat(1, batch_size, 1)

        if warmup:
            fused_feature = torch.concat([cls_tokens, vis_feature], 0)
            fused_feature = self.multi_trans_pre(fused_feature)
        else:
            fused_feature = torch.concat([cls_tokens, convers_feature, vis_feature], 0)
            fused_feature = self.multi_trans(fused_feature)

        logits = self.classifier(fused_feature[0, :, :])
        return logits
