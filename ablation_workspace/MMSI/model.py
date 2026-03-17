import copy
import math

import torch
import torch.nn as nn


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
        batch_size=x.size(1)
        pos_enc = self.encoding[:x.size(0)].detach().unsqueeze(1)
        pos_enc = torch.tile(pos_enc, (1, batch_size, 1))
        return x + pos_enc


class FiLMLayer(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.gamma = nn.Linear(condition_dim, feature_dim)
        self.beta = nn.Linear(condition_dim, feature_dim)

    def forward(self, x, condition):
        # x: [seq_len, batch, dim], condition: [batch, dim]
        conditioned = self.norm(x)
        gamma = self.gamma(condition).unsqueeze(0)
        beta = self.beta(condition).unsqueeze(0)
        return conditioned * (1.0 + gamma) + beta


class BottleneckFusionLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=1024):
        super().__init__()
        self.lang_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.visual_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )

    def forward(self, lang_tokens, visual_tokens, bottleneck_tokens):
        bottleneck_count = bottleneck_tokens.size(0)

        lang_out = self.lang_layer(torch.cat([lang_tokens, bottleneck_tokens], dim=0))
        visual_out = self.visual_layer(torch.cat([visual_tokens, bottleneck_tokens], dim=0))

        shared_bottlenecks = 0.5 * (lang_out[-bottleneck_count:] + visual_out[-bottleneck_count:])
        return lang_out[:-bottleneck_count], visual_out[:-bottleneck_count], shared_bottlenecks


class BottleneckFusionStack(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=1024, num_layers=2,
                 num_bottleneck_layers=1, num_bottleneck_tokens=4):
        super().__init__()
        num_bottleneck_layers = max(1, min(num_layers, num_bottleneck_layers))
        num_prefusion_layers = num_layers - num_bottleneck_layers

        base_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
        )
        self.lang_prefusion_layers = nn.ModuleList(
            [copy.deepcopy(base_layer) for _ in range(num_prefusion_layers)]
        )
        self.visual_prefusion_layers = nn.ModuleList(
            [copy.deepcopy(base_layer) for _ in range(num_prefusion_layers)]
        )
        self.bottleneck_layers = nn.ModuleList(
            [
                BottleneckFusionLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                )
                for _ in range(num_bottleneck_layers)
            ]
        )
        self.bottleneck_tokens = nn.Parameter(torch.randn(num_bottleneck_tokens, 1, d_model))

    def forward(self, lang_tokens, visual_tokens):
        batch_size = lang_tokens.size(1)
        for layer in self.lang_prefusion_layers:
            lang_tokens = layer(lang_tokens)
        for layer in self.visual_prefusion_layers:
            visual_tokens = layer(visual_tokens)

        bottlenecks = self.bottleneck_tokens.repeat(1, batch_size, 1)
        for layer in self.bottleneck_layers:
            lang_tokens, visual_tokens, bottlenecks = layer(lang_tokens, visual_tokens, bottlenecks)

        return torch.cat([lang_tokens, visual_tokens], dim=0)


class MultimodalBaseline(nn.Module):
    def __init__(self, class_num, language_model, visual_film_layers=0, fusion_film_layers=0,
                 projection_dim=128, fusion_mode='standard', bottleneck_tokens=4,
                 bottleneck_fusion_layers=1):
        super(MultimodalBaseline, self).__init__()

        self.class_num = class_num
        self.language_model = language_model
        self.visual_film_layers = visual_film_layers
        self.fusion_film_layers = fusion_film_layers
        self.projection_dim = projection_dim
        self.fusion_mode = fusion_mode
        self.bottleneck_tokens = bottleneck_tokens
        self.bottleneck_fusion_layers = bottleneck_fusion_layers

        if language_model == 'bert':
            from transformers import BertModel
            self.convers_encoder = BertModel.from_pretrained('bert-base-uncased')
        elif language_model == 'roberta':
            from transformers import RobertaModel
            self.convers_encoder = RobertaModel.from_pretrained('roberta-base')
        elif language_model == 'electra':
            from transformers import ElectraModel
            self.convers_encoder = ElectraModel.from_pretrained('google/electra-base-discriminator')
        else:
            raise ValueError(f"Unsupported language model: {language_model}")

        self.convers_fc = nn.Sequential(
            nn.Linear(768, 512))

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
        self.multi_trans_bottleneck = BottleneckFusionStack(
            d_model=512,
            nhead=8,
            dim_feedforward=1024,
            num_layers=2,
            num_bottleneck_layers=bottleneck_fusion_layers,
            num_bottleneck_tokens=bottleneck_tokens,
        )

        self.visual_film = nn.ModuleList(
            [FiLMLayer(feature_dim=512, condition_dim=512) for _ in range(visual_film_layers)]
        )
        self.fusion_film = nn.ModuleList(
            [FiLMLayer(feature_dim=512, condition_dim=512) for _ in range(fusion_film_layers)]
        )

        self.positional_enc = PositionalEncoding(d_model=512, max_len=20)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))

        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, class_num))

    def forward(self, language_token, mask_idxs, keypoint_seqs, speaker_labels, warmup=False, return_features=False):

        # encode language conversation
        pad_val = 0 if self.language_model in ['bert', 'electra'] else 1
        attention_mask = (language_token != pad_val).float()
        convers_festures = self.convers_encoder(language_token, attention_mask)[0]
        convers_feature = convers_festures[torch.arange(len(language_token)), mask_idxs]  # mask token feature
        convers_feature = self.convers_fc(convers_feature)
        convers_feature_token = convers_feature.unsqueeze(0)

        # encode speaker non-verbal behaviors
        batch_size = speaker_labels.size(0)
        gaze_feature, gesture_feature = [], []
        for batch_i in range(batch_size):
            gaze_feature.append(keypoint_seqs[batch_i:batch_i + 1, speaker_labels[batch_i], :, 0:3*2])
            gesture_feature.append(keypoint_seqs[batch_i:batch_i + 1, speaker_labels[batch_i], :, 5*2:11*2])

        speaker_feature = torch.concat([torch.concat(gaze_feature, dim=0), torch.concat(gesture_feature, dim=0)], dim=-1)
        speaker_feature = speaker_feature.view(batch_size, 16, -1, 2)
        speaker_feature = self.coordinate_fc(speaker_feature).view(batch_size, 16, -1)
        speaker_feature = self.speaker_encoder(speaker_feature).permute(1, 0, 2)

        # encode listener positions
        speaker_onehot = torch.nn.functional.one_hot(speaker_labels, num_classes = self.class_num).float()
        speaker_onehot_feature = self.onehot_encoder(speaker_onehot)

        position_feature = keypoint_seqs[:, :, 5, 0:2]
        position_feature = self.coordinate_fc(position_feature).view(batch_size, -1)
        position_feature = self.position_encoder(position_feature) + speaker_onehot_feature
        position_feature = self.position_fc(position_feature).unsqueeze(0)

        # encode visual interactions
        cls_tokens = self.cls_token.repeat(1, batch_size, 1)
        vis_feature = torch.concat([position_feature, self.positional_enc(speaker_feature[::2, :, :])], 0)
        vis_feature = self.visual_trans(vis_feature)
        if not warmup:
            for film_layer in self.visual_film:
                vis_feature = film_layer(vis_feature, convers_feature)
        vis_feature = self.positional_enc(vis_feature)

        # fuse multimodal features
        if warmup: # visual warmup
            fused_feature = torch.concat([cls_tokens, vis_feature], 0)
            fused_feature = self.multi_trans_pre(fused_feature)
        else:
            lang_tokens = torch.concat([cls_tokens, convers_feature_token], 0)
            if self.fusion_mode == 'bottleneck':
                fused_feature = self.multi_trans_bottleneck(lang_tokens, vis_feature)
            else:
                fused_feature = torch.concat([lang_tokens, vis_feature], 0)
                fused_feature = self.multi_trans(fused_feature)
            for film_layer in self.fusion_film:
                fused_feature = film_layer(fused_feature, convers_feature)

        pooled_feature = fused_feature[0, :, :]
        logits = self.classifier(pooled_feature)
        projection = torch.nn.functional.normalize(self.projection_head(pooled_feature), dim=-1)

        if return_features:
            return logits, pooled_feature, projection
        return logits
