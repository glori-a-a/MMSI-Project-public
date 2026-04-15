import copy
import json
import os
import re

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from video_processor import build_temporal_indices, load_frame_sequence


def collate_fn(tokenizer, batch):
    tokens, token_positions, keypoint_seqs, speaker_labels, task_labels, visual_frames = zip(*batch)

    tokens = [torch.tensor(t) for t in tokens]
    tokens = pad_sequence(tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
    token_positions = torch.tensor(token_positions)
    task_labels = torch.tensor(task_labels)
    keypoint_seqs = torch.tensor(np.array(keypoint_seqs)).float()
    speaker_labels = torch.tensor(speaker_labels)

    if visual_frames[0] is None:
        visual_frames = None
    else:
        visual_frames = torch.tensor(np.stack(visual_frames, axis=0)).float()

    return tokens, token_positions, keypoint_seqs, speaker_labels, task_labels, visual_frames


class SocialDataset(Dataset):
    def __init__(self, args, is_training=True):
        self.is_training = is_training
        self.tokenizer = args.tokenizer
        self.language_model = args.language_model
        self.context_length = args.context_length
        self.txt_dir = args.txt_dir
        self.txt_labeled_dir = args.txt_labeled_dir
        self.meta_dir = args.meta_dir
        self.keypoint_dir = args.keypoint_dir
        self.task = args.task
        self.sequence_length = args.sequence_length
        self.video_dir = getattr(args, 'video_dir', None)
        self.visual_feature_type = getattr(args, 'visual_feature_type', 'keypoint')
        self.use_video_frames = self.visual_feature_type in ['vit', 'keypoint_vit', 'marlin', 'keypoint_marlin']
        self.video_fps = getattr(args, 'video_fps', 5)
        self.frame_size = getattr(args, 'frame_size', 224)
        self.alignment_records = []

        with open(args.data_split_file, 'r') as f:
            data_split = json.load(f)
        self.file_names = self.get_file_names(args, data_split, is_training)
        self.mask_token = self.get_mask_token(self.language_model)
        self.start_token_id = self.get_start_token_id()
        self.sep_token_id = self.get_separator_token_id()
        self.data_points = self.load_files(self.file_names)
        max_samples = getattr(args, 'max_train_samples', None) if is_training else getattr(args, 'max_test_samples', None)
        if max_samples is not None and max_samples > 0:
            self.data_points = self.data_points[:max_samples]

    def get_mask_token(self, language_model):
        if language_model in ['bert', 'electra']:
            return '[MASK]'
        if language_model == 'roberta':
            return '<mask>'
        if language_model in ['gpt2', 'bart']:
            return self.tokenizer.mask_token
        raise ValueError(f"Unsupported language model: {language_model}")

    def get_start_token_id(self):
        for token_id in [self.tokenizer.cls_token_id, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]:
            if token_id is not None:
                return token_id
        raise ValueError(f"Tokenizer for {self.language_model} does not define a valid start token")

    def get_separator_token_id(self):
        for token_id in [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id]:
            if token_id is not None:
                return token_id
        raise ValueError(f"Tokenizer for {self.language_model} does not define a valid separator token")

    def get_file_names(self, args, data_split, is_training):
        data_type = 'train' if is_training else 'test'
        return [f"{args.txt_dir}/{file_name}.txt" for file_name in data_split[data_type]]

    def load_files(self, txt_files):
        data_points = []

        with open(f"{self.keypoint_dir}/reference_timestamps.json", 'r') as f:
            reference_timestamps = json.load(f)

        for txt_file in txt_files:
            file_name = txt_file.split('/')[-1].split('.txt')[0]
            txt_file_labeled = f"{self.txt_labeled_dir}/{file_name}.txt"
            keypoint_file = f"{self.keypoint_dir}/{file_name}.npy"
            meta_file = f"{self.meta_dir}/{file_name.split('_')[0]}.json"

            with open(txt_file, 'r') as f:
                utterances = [utterance for utterance in f.read().split('\n') if utterance]

            with open(txt_file_labeled, 'r') as f:
                utterances_labeled = [utterance_labeled for utterance_labeled in f.read().split('\n') if utterance_labeled]

            keypoint_data = np.load(keypoint_file, allow_pickle=True)

            with open(meta_file, 'r') as f:
                meta_data = json.load(f)
            player_num = len(meta_data[file_name.split('_')[1]]['playerNames'])

            reference_timestamp = reference_timestamps[file_name]
            keypoint_seq_ref = self.get_reference_keypoints(keypoint_data[reference_timestamp], player_num)

            data_points.extend(
                self.process_utterances(
                    file_name,
                    utterances,
                    utterances_labeled,
                    keypoint_data,
                    keypoint_seq_ref,
                    player_num,
                )
            )

        return data_points

    def get_reference_keypoints(self, keypoint_data_ref, player_num):
        keypoint_seq_ref = np.zeros((6, 17 * 2))
        for keypoint_ref in keypoint_data_ref:
            if keypoint_ref['idx'] < player_num:
                keypoint_wo_conf = np.delete(keypoint_ref['keypoints'], np.arange(2, len(keypoint_ref['keypoints']), 3))
                keypoint_seq_ref[keypoint_ref['idx'], :] = keypoint_wo_conf[:17 * 2]
        return keypoint_seq_ref

    def process_utterances(self, file_name, utterances, utterances_labeled, keypoint_data, keypoint_seq_ref, player_num):
        data_points = []

        for utterance_i, (utterance, utterance_labeled) in enumerate(zip(utterances, utterances_labeled)):
            time_sec = self.get_time_in_seconds(utterance)
            words = utterance.split()

            start_idx = max(0, utterance_i - self.context_length - max(0, self.context_length - (len(utterances) - utterance_i) + 1))
            end_idx = start_idx + 2 * self.context_length + 1

            is_player_speaker = words[0].startswith('[Player')
            speaker_label = None
            if is_player_speaker:
                speaker_label = int(re.search(r'\[Player(\d+)\]', words[0]).group(1))

            utterance_involved = False
            for word_i, word in enumerate(words):
                data_point = self.process_word(word, word_i, utterance_labeled, words, player_num, is_player_speaker, utterance_involved)
                if data_point:
                    keypoint_seq, keypoint_indices = self.get_keypoint_sequence(
                        keypoint_data,
                        time_sec,
                        player_num,
                        keypoint_seq_ref,
                        speaker_label,
                    )
                    convers_context = self.get_conversation_context(utterances, utterance_i, start_idx, end_idx, data_point[1])
                    masked_i = utterance_i - start_idx
                    self.alignment_records.append({
                        'file_name': file_name,
                        'utterance_index': utterance_i,
                        'time_sec': time_sec,
                        'keypoint_indices': keypoint_indices,
                        'frame_indices': keypoint_indices,
                    })
                    data_points.append(
                        (convers_context, masked_i, keypoint_seq, file_name, time_sec, player_num, speaker_label, data_point[0])
                    )
                    utterance_involved = data_point[2]

        return data_points

    def get_time_in_seconds(self, utterance):
        time_str = utterance.split(': ')[0].split('(')[1][:-1]
        m, s = map(int, time_str.split(':'))
        return m * 60 + s

    def process_word(self, word, word_i, utterance_labeled, words, player_num, is_player_speaker, utterance_involved):
        second_pronouns = ['you', 'your']
        third_pronouns = ['he', 'his', 'him', 'she', 'her']

        if self.task == 'STI' and word.lower() in second_pronouns and utterance_labeled.rstrip().endswith(']') \
                and word_i != 0 and is_player_speaker and not utterance_involved:
            brackets = re.findall(r"\[(.*?)\]", utterance_labeled)
            names_in_bracket = [name in [f'Player{i}' for i in range(player_num)] for name in brackets[-1].split()]
            if names_in_bracket.count(True) == 1:
                words_cp = copy.deepcopy(words)
                task_label = int(re.search(r'Player(\d+)', brackets[-1]).group(1))
                words_cp.append(f'(To {self.mask_token})')
                utterance_involved = True
                return task_label, words_cp, utterance_involved

        elif self.task == 'PCR' and any(pronoun in word.lower() for pronoun in third_pronouns) and \
                ('Player' in utterance_labeled.split()[word_i]) and word_i != 0 and is_player_speaker:
            words_cp = copy.deepcopy(words)
            task_label = int(re.search(r'Player(\d+)', utterance_labeled.split()[word_i]).group(1))
            target_pronoun = [pronoun for pronoun in third_pronouns if pronoun in word.lower()][-1]
            words_cp[word_i] = re.sub(rf'{target_pronoun}', self.mask_token, words_cp[word_i].lower())
            return task_label, words_cp, utterance_involved

        elif self.task == 'MPP' and word.startswith('[Player') and word_i != 0 and is_player_speaker:
            words_cp = copy.deepcopy(words)
            task_label = int(re.search(r'\[Player(\d+)\]', words_cp[word_i]).group(1))
            words_cp[word_i] = re.sub(r'\[Player\d+\]', self.mask_token, words_cp[word_i])
            return task_label, words_cp, utterance_involved

        return None

    def get_keypoint_sequence(self, keypoint_data, time_sec, player_num, keypoint_seq_ref, speaker_label):
        keypoint_seq = np.zeros((6, self.sequence_length, 17 * 2))
        keypoint_indices = build_temporal_indices(
            time_sec=time_sec,
            sequence_length=self.sequence_length,
            fps=self.video_fps,
            total_steps=len(keypoint_data),
        )

        for time_i, time_step in enumerate(keypoint_indices):
            keypoints = keypoint_data[time_step]
            for keypoint in keypoints:
                if keypoint['idx'] < player_num:
                    keypoint_wo_conf = np.delete(keypoint['keypoints'], np.arange(2, len(keypoint['keypoints']), 3))
                    keypoint_seq[keypoint['idx'], time_i, :] = keypoint_wo_conf[:17 * 2]

            for player_i in range(player_num):
                if np.sum(keypoint_seq[player_i, time_i, :]) == 0:
                    keypoint_seq[player_i, time_i, :] = keypoint_seq_ref[player_i]

        zero_indices = np.where(keypoint_seq == 0)
        keypoint_seq = keypoint_seq - np.tile(keypoint_seq[speaker_label:speaker_label + 1][:, :, 0:2], (1, 1, 17))
        keypoint_seq[zero_indices] = 0

        return keypoint_seq, keypoint_indices

    def get_conversation_context(self, utterances, utterance_i, start_idx, end_idx, words_cp):
        target_utterance = ' '.join(words_cp)
        utterances_cp = copy.deepcopy(utterances)
        utterances_cp[utterance_i] = target_utterance
        convers_context = utterances_cp[start_idx:end_idx]
        convers_context = [re.sub(r' \(\d{2}:\d{2}\)', '', utterance) for utterance in convers_context]

        return convers_context

    def apply_augmentation(self, convers_context, keypoint_seq, player_num, speaker_label, task_label):
        if np.random.random() < 0.5:
            keypoint_seq[:, :, ::2] = -1.0 * keypoint_seq[:, :, ::2]
            keypoint_seq_cp = copy.deepcopy(keypoint_seq)
            for change_i in [1, 3, 5, 7, 9]:
                keypoint_seq[:, :, 2 * change_i:2 * change_i + 1] = keypoint_seq_cp[:, :, 2 * change_i + 1:2 * change_i + 2]
                keypoint_seq[:, :, 2 * change_i + 1:2 * change_i + 2] = keypoint_seq_cp[:, :, 2 * change_i:2 * change_i + 1]

        player_numbers = list(range(player_num))
        shuffled_player_numbers = copy.deepcopy(player_numbers)
        np.random.shuffle(shuffled_player_numbers)
        player_number_mapping = {old: new for old, new in zip(player_numbers, shuffled_player_numbers)}

        task_label = player_number_mapping[task_label]
        speaker_label = player_number_mapping[speaker_label]

        convers_context = [re.sub(r'\[Player(\d+)\]', lambda match: '[Player{}]'.format(player_number_mapping[int(match.group(1))]),
                   utterance) for utterance in convers_context]

        inverse_shuffle = np.argsort(shuffled_player_numbers + list(range(player_num, 6)))
        keypoint_seq = keypoint_seq[inverse_shuffle, :, :]

        return convers_context, keypoint_seq, speaker_label, task_label

    def tokenize_conversation(self, convers_context, masked_i):
        tokens = self.tokenizer.encode(convers_context[masked_i], add_special_tokens=False) + [self.sep_token_id]

        i = 0
        while True:
            before_exists = masked_i - i - 1 >= 0
            after_exists = masked_i + i + 1 < len(convers_context)
            before_tokens = self.tokenizer.encode(convers_context[masked_i - i - 1], add_special_tokens=False) if before_exists else []
            after_tokens = self.tokenizer.encode(convers_context[masked_i + i + 1], add_special_tokens=False) if after_exists else []
            if before_exists and len(tokens) + len(before_tokens) + 1 <= 511:
                tokens = before_tokens + [self.sep_token_id] + tokens
            if after_exists and len(tokens) + len(after_tokens) + 1 <= 511:
                tokens = tokens + after_tokens + [self.sep_token_id]
            if not before_exists and not after_exists:
                break
            i += 1

        return [self.start_token_id] + tokens

    def get_visual_frames(self, file_name, time_sec):
        if not self.use_video_frames:
            return None
        if not self.video_dir or self.video_dir == 'enter_the_path':
            raise ValueError(
                f"visual_feature_type={self.visual_feature_type} requires a valid --video_dir or precomputed visual directory"
            )

        cached_clip_path = os.path.join(self.video_dir, f"{file_name}__{int(time_sec)}.npy")
        if os.path.exists(cached_clip_path):
            return np.load(cached_clip_path)

        frames, frame_indices = load_frame_sequence(
            self.video_dir,
            file_name=file_name,
            time_sec=time_sec,
            sequence_length=self.sequence_length,
            fps=self.video_fps,
            frame_size=self.frame_size,
        )
        self.alignment_records.append({
            'file_name': file_name,
            'time_sec': time_sec,
            'frame_indices': frame_indices,
        })
        return frames

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        convers_context, masked_i, keypoint_seq, file_name, time_sec, player_num, speaker_label, task_label = self.data_points[idx]

        if self.is_training:
            convers_context, keypoint_seq, speaker_label, task_label = \
                self.apply_augmentation(convers_context, keypoint_seq, player_num, speaker_label, task_label)

        tokens = self.tokenize_conversation(convers_context, masked_i)
        token_position = tokens.index(self.tokenizer.mask_token_id)
        visual_frames = self.get_visual_frames(file_name, time_sec)

        return tokens, token_position, keypoint_seq, speaker_label, task_label, visual_frames
