#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Model Implementation

@Author  :   Ma (Ma787639046@outlook.com)
'''
import os
import math
import copy
import warnings
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertLayer
from transformers.modeling_outputs import MaskedLMOutput
from arguments import DataTrainingArguments, ModelArguments, PUNRPreTrainingArguments as TrainingArguments
from transformers.modeling_utils import Conv1D, PretrainedConfig
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

# Import BERT cat
from bert_cat import BertForMaskedLM, BertEmbeddings, BertModel

import logging
logger = logging.getLogger(__name__)

class AttentionPooling(nn.Module):
    def __init__(self, emb_size=768, hidden_size=768):
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(emb_size, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x, attn_mask=None):
        """
        Args:
            x: batch_size, candidate_size, emb_dim
            attn_mask: batch_size, candidate_size
        Returns:
            (shape) batch_size, emb_dim
        """
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)
        alpha = torch.exp(alpha)

        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)
        x = torch.bmm(x.permute(0, 2, 1), alpha).squeeze(dim=-1)
        return x

@dataclass
class MaskedLMOutputWithLogs(MaskedLMOutput):
    logs: Optional[Dict[str, any]] = None

class BertForPUNR(nn.Module):
    def __init__(
        self,
        bert: BertModel,
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        train_args: TrainingArguments,
    ):
        super(BertForPUNR, self).__init__()
        self.lm = bert
        self.config = bert.config
        self.cross_entropy = nn.CrossEntropyLoss()
        self.model_args: ModelArguments = model_args
        self.train_args: TrainingArguments = train_args
        self.data_args: DataTrainingArguments = data_args

        # Init a normal transformers-encoder based head
        if model_args.use_enc_head:
            if model_args.init_enc_from_bert:
                ae_dec_init_idxs = list(range(bert.config.num_hidden_layers-model_args.n_enc_head_layers, bert.config.num_hidden_layers))
                logging.info(f"Init AE-Dec param from BERT layer {ae_dec_init_idxs}")
                self.c_head = nn.ModuleList(
                    [copy.deepcopy(bert.bert.encoder.layer[i]) for i in ae_dec_init_idxs]
                )
            else:
                self.c_head = nn.ModuleList(
                    [BertLayer(bert.config) for _ in range(model_args.n_enc_head_layers)]
                )
                self.c_head.apply(self._init_weights)
        
        # Init a transformers-decoder based GPT2 Blocks as head 
        if model_args.use_dec_head:
            self.d_head, self.d_head_drop, self.d_head_ln_f = self.build_ar_dec_head(bert.config, data_args, model_args)
            self.d_head.apply(self._init_weights)
        
        if self.model_args.pooling_strategy == 'att' or self.model_args.pooling_strategy == 'attn':
            self.attention_pooler = AttentionPooling(768, 768)
        else:
            self.attention_pooler = None

    @staticmethod
    def build_ar_dec_head(config, data_args, model_args):
        # Cast a GPT2Config from BERTConfig
        gpt2_config = GPT2Config(
            vocab_size=config.vocab_size, 
            n_positions=config.max_position_embeddings,
            n_embd=config.hidden_size,
            n_layer=config.num_hidden_layers,
            n_head=config.num_attention_heads,
            activation_function=config.hidden_act,
            resid_pdrop=config.hidden_dropout_prob,
            embd_pdrop=config.hidden_dropout_prob,
            attn_pdrop=config.attention_probs_dropout_prob,
            layer_norm_epsilon=config.layer_norm_eps,
            initializer_range=config.initializer_range,
        )
        ar_dec_head = nn.ModuleList(
            [GPT2Block(gpt2_config, layer_idx=i) for i in range(model_args.n_dec_head_layers)]
        )
        # Adjust to Prefix Attn Mask if `_prompt_input_ids` in data_args
        if hasattr(data_args, 'prefix_width'):
            assert data_args.prefix_width >= 1
            # get diag ones
            casual_attn_mask = torch.tril(torch.ones((gpt2_config.n_positions, gpt2_config.n_positions), dtype=torch.uint8)).view(
                                                1, 1, gpt2_config.n_positions, gpt2_config.n_positions)
            # Customized casual attention mask with `prefix_width`
            for i in range(0, casual_attn_mask.size(-2)):   # row [0, n_positions]
                casual_attn_mask[0][0][i][:data_args.prefix_width+1] = 1
            _block: GPT2Block = None
            for _block in ar_dec_head:
                del _block.attn.bias
                _block.attn.register_buffer("bias", casual_attn_mask)

        if model_args.attn_window != -1:
            casual_attn_mask = torch.tril(torch.ones((gpt2_config.n_positions, gpt2_config.n_positions), dtype=torch.uint8)).view(
                                                1, 1, gpt2_config.n_positions, gpt2_config.n_positions
                                            )
            # Customized casual attention mask, attention only on cls & tokens within model_args.attn_window
            for i in range(model_args.attn_window + 1, casual_attn_mask.size(-2)):   # row start from model_args.attn_window + 1
                for j in range(1, i-model_args.attn_window+1):    # col start from 1
                    casual_attn_mask[0][0][i][j] = 0
            assert model_args.attn_window > 0
            _block: GPT2Block = None
            for _block in ar_dec_head:
                del _block.attn.bias
                _block.attn.register_buffer("bias", casual_attn_mask)
        
        # A Dropout is applied before embedding going into model in GPT2 settings
        ar_dec_head_dropout_layer = nn.Dropout(gpt2_config.embd_pdrop)
        # A LN is applied after output hidden states of GPT2Block in GPT2 settings
        ar_dec_head_ln_layer = nn.LayerNorm(gpt2_config.n_embd, eps=gpt2_config.layer_norm_epsilon)
        
        return ar_dec_head, ar_dec_head_dropout_layer, ar_dec_head_ln_layer
    
    def pooling(self,
                last_hidden: Tuple[torch.Tensor]=None, 
                attention_mask: torch.Tensor=None,):
        """
        Poolers to get the sentence embedding
        'cls': [CLS] representation without BERT/RoBERTa's MLP pooler.
        'avg': average of the last layers' hidden states at each token.
        'att': MLP layer for attention pooling
        """
        if self.model_args.pooling_strategy == 'cls':
            return last_hidden[:, 0]
        elif self.model_args.pooling_strategy == 'avg':
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.model_args.pooling_strategy == 'att' or self.model_args.pooling_strategy == 'attn':
            return self.attention_pooler(last_hidden, attention_mask)
        else:
            raise NotImplementedError()

    def forward(self, **model_input):
        with nullcontext() if not self.model_args.freeze_bert else torch.no_grad():
            lm_out: MaskedLMOutput = self.lm.forward(
                input_ids=model_input['input_ids'],
                attention_mask=model_input['attention_mask'],
                news_segment_ids=model_input['news_segment_ids'], 
                labels=model_input['labels'],
                output_hidden_states=True,
                return_dict=True
            )

        # cls_hiddens = lm_out.hidden_states[-1][:, :1]
        cls_hiddens = self.pooling(lm_out.hidden_states[-1], model_input['attention_mask'])
        cls_hiddens = cls_hiddens.unsqueeze(1)

        logs = dict()

        # add last layer mlm loss
        loss = 0.0
        if not self.model_args.disable_bert_mlm_loss:
            loss = lm_out.loss
            logs["bert_mlm_loss"] = lm_out.loss.item()
        
        if self.model_args.use_enc_head:
            # Get the embedding of decoder inputs
            enc_head_emb = self.lm.bert.embeddings(
                input_ids=model_input['ae_dec_head_input_ids'],
                news_segment_ids=model_input['ae_dec_head_news_segment_ids'],
            )
            enc_head_attn_mask = self.lm.get_extended_attention_mask(
                                        model_input['ae_dec_head_attention_mask'],
                                        model_input['ae_dec_head_attention_mask'].shape,
                                        model_input['ae_dec_head_attention_mask'].device
                                    )
            # Concat cls-hiddens of span A & embedding of span B
            c_head_hiddens = torch.cat([cls_hiddens, enc_head_emb[:, 1:]], dim=1)
            for layer in self.c_head:
                layer_out = layer(
                    c_head_hiddens,
                    enc_head_attn_mask,
                )
                c_head_hiddens = layer_out[0]
            
            if self.model_args.enable_enc_head_mlm:
                # add head-layer mlm loss
                enc_head_mlm_loss = self.mlm_loss(c_head_hiddens, model_input['ae_dec_head_labels']) * self.model_args.enc_head_mlm_coef
                logs["enc_head_mlm_loss"] = enc_head_mlm_loss.item()
                loss += enc_head_mlm_loss

        if self.model_args.use_dec_head:
            dec_head_emb = self.lm.bert.embeddings(
                input_ids=model_input['ar_dec_head_input_ids'],
                news_segment_ids=model_input['ar_dec_head_news_segment_ids']
            )
            dec_head_attn_mask = self.lm.get_extended_attention_mask(
                                        model_input['ar_dec_head_attention_mask'],
                                        model_input['ar_dec_head_attention_mask'].shape,
                                        model_input['ar_dec_head_attention_mask'].device
                                    )
            # d_head_hiddens: [bz, tgt_len, hid], (CLS + tgt emb[1:]) at dim 1
            d_head_hiddens = torch.cat([cls_hiddens, dec_head_emb[:, 1:]], dim=1)
            d_head_hiddens = self.d_head_drop(d_head_hiddens) # Dropout on top of embeddings, following GPT2
            for layer in self.d_head:
                layer_out = layer(
                    d_head_hiddens,
                    attention_mask=dec_head_attn_mask,
                )
                d_head_hiddens = layer_out[0]
            d_head_hiddens = self.d_head_ln_f(d_head_hiddens) # LN after hidden states, following GPT2

            if self.model_args.enable_dec_head_loss:
                # labels for generative AR-Dec head is natually its input ids sliced with [..., 1:, :]
                # we will do slice (or logits shift) inside the func `casual_loss`
                # Here we fill the labels (or input ids of AR-Dec) [PAD] area with -100, to avoid loss calculatation
                # on [PAD] area by CrossEntropy loss function
                # Ignore klloss on [PAD]:
                # [cls] xxx xxx xxx [SEP] [PAD] [PAD]
                #   0    0   0   0    0   -100  -100
                # Size [bs, seq_len]
                dec_head_labels = model_input['ar_dec_head_labels']
                # add decoder head layer loss
                dec_head_casl_loss = self.casual_loss(d_head_hiddens, dec_head_labels) \
                                        * self.model_args.dec_head_coef
                logs["dec_head_casual_loss"] = dec_head_casl_loss.item()
                loss += dec_head_casl_loss

        return MaskedLMOutputWithLogs(
            loss=loss,
            logits=lm_out.logits,
            hidden_states=lm_out.hidden_states,
            attentions=lm_out.attentions,
            logs=logs,
        )

    def casual_loss(self, hiddens, labels):
        pred_logits = self.lm.cls(hiddens)
        shift_logits = pred_logits[..., :-1, :].contiguous()    # Only measure the generation between 0~n-1
        shift_labels = labels[..., 1:].contiguous()     # No first token, label=1~n for generation
        loss = self.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        return loss

    def mlm_loss(self, hiddens, labels):
        pred_scores = self.lm.cls(hiddens)
        loss = self.cross_entropy(
            pred_scores.view(-1, self.config.vocab_size),
            labels.view(-1)
        )
        return loss

    def _init_weights(self, module):
        """Initialize the weights. Fetch from GPT2PreTrainedModel"""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if "c_proj" in name and "weight" in name:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.model_args.n_dec_head_layers)))

    @classmethod
    def from_pretrained(
        cls, 
        model_args: ModelArguments,
        data_args: DataTrainingArguments,
        train_args: TrainingArguments,
        *args, **kwargs
    ):
        path = args[0]
        # Load BERT Encoder
        hf_model = BertForMaskedLM.from_pretrained(*args, **kwargs)
        # Init model
        model = cls(hf_model, model_args, data_args, train_args)
        if os.path.exists(os.path.join(path, 'model.pt')):
            warnings.warn('loading extra weights from local files')
            model_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            model.load_state_dict(model_dict, strict=False)
        return model

    @classmethod
    def from_config(
            cls,
            config: PretrainedConfig,
            model_args: ModelArguments,
            data_args: DataTrainingArguments,
            train_args: TrainingArguments,
    ):
        hf_model = BertForMaskedLM.from_config(config)
        model = cls(hf_model, model_args, data_args, train_args)
        return model

    def save_pretrained(self, output_dir: str, *args, **kwargs):
        # Save BERT Encoder
        self.lm.save_pretrained(output_dir, *args, **kwargs)
        # Save head weights
        model_dict = self.state_dict()
        hf_weight_keys = [k for k in model_dict.keys() if k.startswith('lm')]
        warnings.warn(f'Omiting {len(hf_weight_keys)} transformer weights')
        for k in hf_weight_keys:
            model_dict.pop(k)
        torch.save(model_dict, os.path.join(output_dir, 'model.pt'))
        # Save attention pooler if exists
        if self.attention_pooler is not None:
            torch.save(self.attention_pooler.state_dict(), os.path.join(output_dir, 'attn.pt'))
        torch.save([self.data_args, self.model_args, self.train_args], os.path.join(output_dir, 'args.pt'))
    
    @staticmethod
    def _load_extra_weights(model: torch.nn.Module, path: str):
        if os.path.exists(os.path.join(path, 'model.pt')):
            warnings.warn('loading extra weights from local files')
            state_dict = torch.load(os.path.join(path, 'model.pt'), map_location="cpu")
            load_result = model.load_state_dict(state_dict, strict=False)
            # release memory
            del state_dict

    def load(self, *args, **kwargs):
        path = args[0]
        # Load BERT Encoder
        hf_model = BertForMaskedLM.from_pretrained(*args, **kwargs)
        load_results = self.lm.load_state_dict(hf_model.state_dict())
        self._load_extra_weights(self, path)
        # release memory
        del hf_model
    