import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import copy
from typing import Tuple
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig

from transformers import (
    BertTokenizer, 
    BertConfig,
    RobertaTokenizer, 
    RobertaConfig, 
    RobertaModel,
)

# Use Pretrained BERT-cat model for News Encoder & User Encoder
from .bert_cat import BertModel

from .attn import AttentionPooling, ScaledDotProductAttention, MultiHeadSelfAttention

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)
}

class ModelBert(nn.Module):
    def __init__(self, 
                 news_encoder: PreTrainedModel, 
                 user_encoder: PreTrainedModel, 
                 attention_pooler: nn.Module = None,
                 args = None,
                ):
        super(ModelBert, self).__init__()
        self.args = args
        self.pooling_strategy = args.pooling
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder
        self.attention_pooler = attention_pooler
        self.loss_fn = nn.CrossEntropyLoss()

    def pooling(self,
                last_hidden: Tuple[torch.Tensor]=None, 
                attention_mask: torch.Tensor=None,):
        """
        Poolers to get the sentence embedding
        'cls': [CLS] representation without BERT/RoBERTa's MLP pooler.
        'avg': average of the last layers' hidden states at each token.
        'att': MLP layer for attention pooling
        """
        if self.pooling_strategy == 'cls':
            return last_hidden[:, 0]
        elif self.pooling_strategy == 'avg':
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooling_strategy == 'att' or self.pooling_strategy == 'attn':
            return self.attention_pooler(last_hidden, attention_mask)
        else:
            raise NotImplementedError()
    
    def get_news_vec(self, news_inputs):
        news_lm_out = self.news_encoder(**news_inputs)
        news_vec = self.pooling(news_lm_out.last_hidden_state, news_inputs['attention_mask'])
        return news_vec
    
    def get_user_vec(self, user_inputs):
        user_lm_out = self.user_encoder(**user_inputs)
        user_vec = self.pooling(user_lm_out.last_hidden_state, user_inputs['attention_mask'])   # [bs, hs]
        return user_vec

    def forward(self, user_inputs, news_inputs, label=None):
        '''
            user_inputs: batch_size, 512
            news_inputs: batch_size*(1+k), num_word_title
            label: batch_size, 1+K
        '''
        user_vec = self.get_user_vec(user_inputs)   # [bs, hs]
        news_vec = self.get_news_vec(news_inputs)        
        news_vec = news_vec.reshape(user_vec.shape[0], -1, news_vec.shape[-1])  # [bs, 1+k, hs]

        score = torch.bmm(news_vec, user_vec.unsqueeze(dim=-1)).squeeze(dim=-1) # [bs, 1+k]

        loss = 0.0
        if label is not None:
            loss = self.loss_fn(score, label)
        
        return loss, score
    
    @classmethod
    def build(
            cls,
            args,
    ):
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        model_name_or_path = args.model_name
        config: PretrainedConfig = config_class.from_pretrained(model_name_or_path)
        config.update({'user_log_length': args.user_log_length})

        user_encoder = model_class.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
        )
        if args.untied_encoder:
            news_encoder = copy.deepcopy(user_encoder)
        else:
            news_encoder = user_encoder

        attention_pooler = AttentionPooling(emb_size=config.hidden_size, hidden_size=config.hidden_size)
        attn_pooler_path = os.path.join(model_name_or_path, 'attn.pt')
        if os.path.exists(attn_pooler_path):
            warnings.warn(f"Loading attention pooler from {attn_pooler_path}.")
            attn_state_dict = torch.load(attn_pooler_path, map_location='cpu')
            attention_pooler.load_state_dict(attn_state_dict, strict=True)

        return cls(
            news_encoder=news_encoder,
            user_encoder=user_encoder,
            attention_pooler=attention_pooler,
            args=args
        )
        
