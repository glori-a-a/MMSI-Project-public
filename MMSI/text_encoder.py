import torch
import torch.nn as nn


MODEL_SPECS = {
    'bert': {
        'tokenizer_name': 'bert-base-uncased',
        'model_name': 'bert-base-uncased',
        'tokenizer_class': 'BertTokenizer',
        'model_class': 'BertModel',
        'default_pooling': 'mask',
    },
    'roberta': {
        'tokenizer_name': 'roberta-base',
        'model_name': 'roberta-base',
        'tokenizer_class': 'RobertaTokenizer',
        'model_class': 'RobertaModel',
        'default_pooling': 'mask',
    },
    'electra': {
        'tokenizer_name': 'google/electra-base-discriminator',
        'model_name': 'google/electra-base-discriminator',
        'tokenizer_class': 'ElectraTokenizer',
        'model_class': 'ElectraModel',
        'default_pooling': 'mask',
    },
    'gpt2': {
        'tokenizer_name': 'gpt2',
        'model_name': 'gpt2',
        'tokenizer_class': 'GPT2Tokenizer',
        'model_class': 'GPT2Model',
        'default_pooling': 'last',
    },
    'bart': {
        'tokenizer_name': 'facebook/bart-base',
        'model_name': 'facebook/bart-base',
        'tokenizer_class': 'BartTokenizer',
        'model_class': 'BartModel',
        'default_pooling': 'mask',
    },
}


def get_model_spec(language_model):
    if language_model not in MODEL_SPECS:
        raise ValueError(f"Unsupported language model: {language_model}")
    return MODEL_SPECS[language_model]


def get_tokenizer(language_model):
    spec = get_model_spec(language_model)

    if language_model == 'bert':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(spec['tokenizer_name'])
    elif language_model == 'roberta':
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(spec['tokenizer_name'])
    elif language_model == 'electra':
        from transformers import ElectraTokenizer
        tokenizer = ElectraTokenizer.from_pretrained(spec['tokenizer_name'])
    elif language_model == 'gpt2':
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(spec['tokenizer_name'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.mask_token is None:
            tokenizer.add_special_tokens({'mask_token': '<mask>'})
    elif language_model == 'bart':
        from transformers import BartTokenizer
        tokenizer = BartTokenizer.from_pretrained(spec['tokenizer_name'])
    else:
        raise ValueError(f"Unsupported language model: {language_model}")

    return tokenizer


class TextEncoderFactory(nn.Module):
    def __init__(self, language_model, tokenizer, pooling='auto'):
        super().__init__()
        spec = get_model_spec(language_model)

        self.language_model = language_model
        self.pooling = spec['default_pooling'] if pooling == 'auto' else pooling

        if language_model == 'bert':
            from transformers import BertModel
            self.encoder = BertModel.from_pretrained(spec['model_name'])
        elif language_model == 'roberta':
            from transformers import RobertaModel
            self.encoder = RobertaModel.from_pretrained(spec['model_name'])
        elif language_model == 'electra':
            from transformers import ElectraModel
            self.encoder = ElectraModel.from_pretrained(spec['model_name'])
        elif language_model == 'gpt2':
            from transformers import GPT2Model
            self.encoder = GPT2Model.from_pretrained(spec['model_name'])
        elif language_model == 'bart':
            from transformers import BartModel
            self.encoder = BartModel.from_pretrained(spec['model_name'])
        else:
            raise ValueError(f"Unsupported language model: {language_model}")

        if len(tokenizer) != self.encoder.get_input_embeddings().num_embeddings:
            self.encoder.resize_token_embeddings(len(tokenizer))

        self.hidden_size = self.encoder.config.hidden_size
        self.pad_token_id = tokenizer.pad_token_id

    def forward(self, input_ids, attention_mask, token_positions=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state

        if self.pooling == 'mask':
            if token_positions is None:
                raise ValueError("token_positions are required for mask pooling")
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_indices, token_positions]

        if self.pooling == 'last':
            if attention_mask is None:
                raise ValueError("attention_mask is required for last-token pooling")
            token_positions = attention_mask.long().sum(dim=1) - 1
            token_positions = token_positions.clamp_min(0)
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_indices, token_positions]

        if self.pooling == 'mean':
            if attention_mask is None:
                raise ValueError("attention_mask is required for mean pooling")
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            pooled = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1e-6)
            return pooled / denom

        raise ValueError(f"Unsupported pooling strategy: {self.pooling}")
