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


class GraphInteractionLayer(nn.Module):
    def __init__(self, d_model=512, speaker_centered=False, condition_dim=512):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.rel_proj = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.condition_norm = nn.LayerNorm(d_model)
        self.cond_gamma = nn.Linear(condition_dim, d_model)
        self.cond_beta = nn.Linear(condition_dim, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.scale = math.sqrt(d_model)
        self.speaker_centered = speaker_centered
        if speaker_centered:
            self.speaker_bias = nn.Embedding(3, 1)

    def forward(self, node_features, player_centers, player_mask, speaker_labels, condition=None):
        # node_features: [batch, players, dim], player_centers: [batch, players, 2]
        # player_mask: [batch, players], 1 for valid players, 0 otherwise
        if condition is not None:
            conditioned = self.condition_norm(node_features)
            gamma = self.cond_gamma(condition).unsqueeze(1)
            beta = self.cond_beta(condition).unsqueeze(1)
            node_features = conditioned * (1.0 + gamma) + beta

        q = self.query_proj(node_features)
        k = self.key_proj(node_features)
        v = self.value_proj(node_features)

        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / self.scale
        relative_pos = player_centers.unsqueeze(2) - player_centers.unsqueeze(1)
        rel_bias = torch.tanh(self.rel_proj(relative_pos)).squeeze(-1) * 2.0
        attn_logits = attn_logits + rel_bias

        if self.speaker_centered:
            player_ids = torch.arange(node_features.size(1), device=node_features.device).view(1, -1)
            source_is_speaker = player_ids.unsqueeze(2) == speaker_labels.view(-1, 1, 1)
            target_is_speaker = player_ids.unsqueeze(1) == speaker_labels.view(-1, 1, 1)
            edge_type = torch.zeros_like(attn_logits, dtype=torch.long)
            edge_type = torch.where(source_is_speaker, torch.ones_like(edge_type), edge_type)
            edge_type = torch.where(target_is_speaker, torch.full_like(edge_type, 2), edge_type)
            attn_logits = attn_logits + self.speaker_bias(edge_type).squeeze(-1)

        self_mask = torch.eye(node_features.size(1), device=node_features.device).unsqueeze(0)
        valid_targets = player_mask.unsqueeze(1).bool()
        attn_logits = attn_logits.masked_fill(self_mask.bool(), -1e4)
        attn_logits = attn_logits.masked_fill(~valid_targets, -1e4)
        attn_logits = attn_logits - attn_logits.amax(dim=-1, keepdim=True)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        attn_weights = attn_weights * valid_targets.float()

        aggregated = torch.matmul(attn_weights, v)
        aggregated = aggregated * player_mask.unsqueeze(-1)
        node_features = self.norm1(node_features + self.out_proj(aggregated))
        node_features = self.norm2(node_features + self.ffn(node_features))
        node_features = node_features * player_mask.unsqueeze(-1)
        return node_features


class MultimodalBaseline(nn.Module):
    def __init__(self, class_num, language_model, visual_film_layers=0, fusion_film_layers=0,
                 projection_dim=128, fusion_mode='standard', bottleneck_tokens=4,
                 bottleneck_fusion_layers=1, graph_layers=0, graph_mode='standard',
                 classifier_mode='standard'):
        super(MultimodalBaseline, self).__init__()

        self.class_num = class_num
        self.language_model = language_model
        self.visual_film_layers = visual_film_layers
        self.fusion_film_layers = fusion_film_layers
        self.projection_dim = projection_dim
        self.fusion_mode = fusion_mode
        self.bottleneck_tokens = bottleneck_tokens
        self.bottleneck_fusion_layers = bottleneck_fusion_layers
        self.graph_layers = graph_layers
        self.graph_mode = graph_mode
        self.classifier_mode = classifier_mode

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
        self.graph_node_encoder = nn.Sequential(
            nn.Linear(17 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
        )
        self.graph_role_embedding = nn.Embedding(2, 512)
        self.graph_input_norm = nn.LayerNorm(512)
        self.graph_summary_query = nn.Linear(512, 512)
        self.graph_summary_norm = nn.LayerNorm(512)
        self.graph_layers_stack = nn.ModuleList(
            [
                GraphInteractionLayer(
                    d_model=512,
                    speaker_centered=(graph_mode in ['speaker_centered', 'speaker_contextual']),
                    condition_dim=512,
                )
                for _ in range(graph_layers)
            ]
        )

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
        self.relational_scorer = nn.Sequential(
            nn.Linear(512 * 5 + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.jepa_online_proj = nn.Linear(512, 512)
        self.jepa_target_proj = nn.Linear(512, 512)
        self.jepa_target_proj.load_state_dict(self.jepa_online_proj.state_dict())
        for param in self.jepa_target_proj.parameters():
            param.requires_grad = False
        self.jepa_predictor = nn.Sequential(
            nn.Linear(512 * 3 + 64, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

    @torch.no_grad()
    def update_jepa_target(self, momentum=0.99):
        for target_param, online_param in zip(self.jepa_target_proj.parameters(), self.jepa_online_proj.parameters()):
            target_param.data.mul_(momentum).add_(online_param.data, alpha=1.0 - momentum)

    def _compute_jepa_loss(self, player_nodes, player_centers, player_mask, speaker_labels, convers_feature,
                           mask_count=1):
        batch_size, num_players, _ = player_nodes.shape
        device = player_nodes.device
        total_loss = player_nodes.new_tensor(0.0)
        used_targets = 0

        for batch_idx in range(batch_size):
            valid_players = torch.nonzero(player_mask[batch_idx] > 0, as_tuple=False).squeeze(-1)
            non_speaker = valid_players[valid_players != speaker_labels[batch_idx]]
            if non_speaker.numel() == 0:
                continue

            perm = torch.randperm(non_speaker.numel(), device=device)
            target_players = non_speaker[perm[:min(mask_count, non_speaker.numel())]]
            context_mask = player_mask[batch_idx].clone().bool()
            context_mask[target_players] = False
            if context_mask.sum() == 0:
                continue

            context_nodes = player_nodes[batch_idx][context_mask]
            context_summary = context_nodes.mean(dim=0, keepdim=True)
            speaker_node = player_nodes[batch_idx, speaker_labels[batch_idx]].unsqueeze(0)
            language_node = convers_feature[batch_idx].unsqueeze(0)

            relative_centers = player_centers[batch_idx, target_players] - player_centers[batch_idx, speaker_labels[batch_idx]].unsqueeze(0)
            relative_pos = self.coordinate_fc(relative_centers.unsqueeze(0)).squeeze(0).view(target_players.size(0), -1)

            pred_inputs = torch.cat([
                context_summary.expand(target_players.size(0), -1),
                speaker_node.expand(target_players.size(0), -1),
                language_node.expand(target_players.size(0), -1),
                relative_pos,
            ], dim=-1)
            predicted_targets = self.jepa_predictor(pred_inputs)

            online_targets = self.jepa_online_proj(player_nodes[batch_idx, target_players])
            with torch.no_grad():
                target_targets = self.jepa_target_proj(player_nodes[batch_idx, target_players].detach())

            predicted_targets = torch.nn.functional.normalize(predicted_targets, dim=-1)
            target_targets = torch.nn.functional.normalize(target_targets, dim=-1)
            online_targets = torch.nn.functional.normalize(online_targets, dim=-1)

            prediction_loss = torch.nn.functional.mse_loss(predicted_targets, target_targets, reduction='sum')
            alignment_loss = torch.nn.functional.mse_loss(online_targets, target_targets, reduction='sum')
            total_loss = total_loss + prediction_loss + 0.5 * alignment_loss
            used_targets += target_players.size(0)

        if used_targets == 0:
            return player_nodes.new_tensor(0.0)
        return total_loss / used_targets

    def forward(self, language_token, mask_idxs, keypoint_seqs, speaker_labels, warmup=False,
                return_features=False, return_aux=False, compute_jepa=False, jepa_mask_count=1):

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

        graph_tokens = []
        player_nodes = None
        player_centers = None
        player_mask = None
        if self.graph_layers > 0:
            keypoint_players = keypoint_seqs.view(batch_size, self.class_num, 16, 17, 2)
            valid_mask = (keypoint_players.abs().sum(dim=-1) > 0).float().unsqueeze(-1)
            valid_count = valid_mask.sum(dim=2).clamp(min=1.0)
            avg_pose = (keypoint_players * valid_mask).sum(dim=2) / valid_count
            player_centers = avg_pose[:, :, 5, :]
            player_mask = (valid_mask.sum(dim=(2, 3, 4)) > 0).float()
            player_nodes = self.graph_node_encoder(avg_pose.reshape(batch_size, self.class_num, -1))

            player_ids = torch.arange(self.class_num, device=keypoint_seqs.device).unsqueeze(0)
            speaker_roles = (player_ids == speaker_labels.unsqueeze(1)).long()
            player_nodes = player_nodes + self.graph_role_embedding(speaker_roles)
            player_nodes = self.graph_input_norm(player_nodes)
            player_nodes = player_nodes * player_mask.unsqueeze(-1)

            for graph_layer in self.graph_layers_stack:
                player_nodes = graph_layer(
                    player_nodes,
                    player_centers,
                    player_mask,
                    speaker_labels,
                    condition=convers_feature,
                )

            graph_tokens = [player_nodes.permute(1, 0, 2)]
            if self.graph_mode == 'speaker_contextual':
                summary_query = self.graph_summary_query(convers_feature).unsqueeze(1)
                summary_logits = torch.matmul(summary_query, player_nodes.transpose(1, 2)).squeeze(1) / math.sqrt(player_nodes.size(-1))
                summary_logits = summary_logits.masked_fill(~player_mask.bool(), -1e4)
                summary_logits = summary_logits - summary_logits.amax(dim=-1, keepdim=True)
                summary_weights = torch.softmax(summary_logits, dim=-1)
                summary_weights = torch.nan_to_num(summary_weights, nan=0.0, posinf=0.0, neginf=0.0)
                summary_feature = torch.matmul(summary_weights.unsqueeze(1), player_nodes).squeeze(1)
                summary_feature = self.graph_summary_norm(summary_feature + convers_feature)
                graph_tokens.append(summary_feature.unsqueeze(0))

        # encode visual interactions
        cls_tokens = self.cls_token.repeat(1, batch_size, 1)
        vis_feature_parts = [position_feature, self.positional_enc(speaker_feature[::2, :, :])]
        vis_feature_parts.extend(graph_tokens)
        vis_feature = torch.concat(vis_feature_parts, 0)
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
        if self.classifier_mode == 'relational' and player_nodes is not None:
            speaker_nodes = player_nodes[torch.arange(batch_size, device=player_nodes.device), speaker_labels]
            speaker_nodes = speaker_nodes.unsqueeze(1).expand(-1, self.class_num, -1)
            pooled_expand = pooled_feature.unsqueeze(1).expand(-1, self.class_num, -1)
            language_expand = convers_feature.unsqueeze(1).expand(-1, self.class_num, -1)
            relative_centers = player_centers - player_centers[torch.arange(batch_size, device=player_centers.device), speaker_labels].unsqueeze(1)
            relative_centers = self.coordinate_fc(relative_centers).view(batch_size, self.class_num, -1)
            relation_feature = torch.cat(
                [
                    player_nodes,
                    speaker_nodes,
                    player_nodes - speaker_nodes,
                    pooled_expand,
                    language_expand,
                    relative_centers,
                ],
                dim=-1,
            )
            logits = self.relational_scorer(relation_feature).squeeze(-1)
            logits = logits.masked_fill(~player_mask.bool(), -1e4)
        else:
            logits = self.classifier(pooled_feature)
        projection = torch.nn.functional.normalize(self.projection_head(pooled_feature), dim=-1)
        aux_outputs = {}
        if compute_jepa and player_nodes is not None:
            aux_outputs['jepa_loss'] = self._compute_jepa_loss(
                player_nodes,
                player_centers,
                player_mask,
                speaker_labels,
                convers_feature,
                mask_count=jepa_mask_count,
            )

        if return_features and return_aux:
            return logits, pooled_feature, projection, aux_outputs
        if return_features:
            return logits, pooled_feature, projection
        return logits
