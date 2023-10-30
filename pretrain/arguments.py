#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Training Arguments

@Author  :   Ma (Ma787639046@outlook.com)
'''
from dataclasses import dataclass, field
from typing import Optional, Union
import os
from transformers import TrainingArguments

@dataclass
class DataTrainingArguments:
    """
    Arguments control input data path, mask behaviors
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: str = field(
        default=None, metadata={"help": "Path to train data"}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated. Default to the max input length of the model."
        },
    )
    min_seq_length: int = field(default=16)
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    data_type: str = field(
        default='title',
        metadata={
            "help": "Choose between 'title', 'abstract'."
        },
    )
    sample_from_spans: bool = field(
        default=False, 
        metadata={"help": 
                  "Whether to sample from a bunch of spans from same document."
                  "This ensures no spans from same document appear in one batch."
                  "Useful when you want to add contrastive learning object to PUNR."
        }
    )
    bert_mask_ratio: float = field(
        default=0.30, metadata={"help": "Ratio of tokens to mask for BERT"}
    )
    enc_head_mask_ratio: float = field(
        default=0.45, metadata={"help": "Ratio of tokens to mask for Transformers-encoder based head"}
    )
    add_prompt_prefix: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to add 'title:', 'abstract:' prompt prefix text in front of each input."}
    )

    content_window_length_ratio: float = field(
        default=0.5, metadata={"help": "How many tokens on left & right in total for emb stream to attend the content stream"}
    )

    user_log_length: int = field(
        default=50, metadata={"help": "Num of user log spans"}
    )

    # Mask entire history spans
    # The masked entire span is reconstructed with other unmasked / partially masked history spans
    # For user level modeling
    mask_whole_spans: Optional[bool] = field(
        default=False,
        metadata={"help": "Mask whole news title spans of user history. For user modeling."}
    )
    mask_whole_spans_token_ratio: float = field(
        default=0.8, metadata={"help": "Ratio for masking whole spans."}
    )

    def __post_init__(self):
        if self.train_dir is not None:
            files = os.listdir(self.train_dir)
            self.train_path = [
                os.path.join(self.train_dir, f)
                for f in files
                if f.endswith('tsv') or f.endswith('json')
            ]
        if '+' in self.data_type:
            _data_types = self.data_type.split('+')
            self.data_type = [i.strip() for i in _data_types]

@dataclass
class ModelArguments:
    """
    Arguments control model config, decoder head config
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default='bert',
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    disable_bert_mlm_loss: bool = field(
        default=False,
        metadata={"help": "Whether to disable BERT MLM loss"},
    )
    freeze_bert: bool = field(
        default=False,
        metadata={"help": "Whether to disable BERT Grad"},
    )

    pooling_strategy: str = field(
        default='cls',
        metadata={"help": "Choose between 'cls', 'avg', 'attn'."},
    )

    # Transformers-Encoder-based head mlm
    n_enc_head_layers: int = field(default=1)
    use_enc_head: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to use a transformer-encoder head of MAE, please set to True"}
    )
    init_enc_from_bert: Optional[bool] = field(
        default=False,
        metadata={"help": "Init Enc from last layers of BERT."}
    )
    enable_enc_head_mlm: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to add encoder-based-head layer mlm loss"}
    )
    enc_head_mlm_coef: Optional[float] = field(default=1.0)

    
    # Transformers-Decoder-based head clm
    n_dec_head_layers: int = field(default=1)
    use_dec_head: Optional[bool] = field(
        default=False,
        metadata={"help": "If you want to use a transformer-decoder head for autoregression generation, please set to True"}
    )
    enable_dec_head_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to add decode-based-head layer CE loss"}
    )
    dec_head_coef: Optional[float] = field(default=1.0)
    
    # Customized casual attention mask, attention only on cls & tokens within model_args.attn_window
    attn_window: int = field(
        default=-1,
        metadata={"help": "Set a triangle casual attention mask with attention window span restrictions."
                          "-1 to disable this act."
        }
    )

    # UserEncoder
    news_dim: Optional[int] = field(
        default=768,
        metadata={"help": "Projection out dim for News Encoder"}
    )
    num_attention_heads: Optional[int] = field(
        default=12,
        metadata={"help": "Control Nums of Attn Head when uses NRMS model in UserEncoder."}
    )
    hidden_dim: Optional[int] = field(
        default=64,
        metadata={"help": "Control Hidden Dim of each Attn Head when uses NRMS model in UserEncoder."}
    )
    user_query_vector_dim: Optional[int] = field(
        default=768,
        metadata={"help": "Out dim for User Encoder vector"}
    )

@dataclass
class PUNRPreTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    remove_unused_columns: bool = field(default=False)
    logging_path: Optional[str] = field(
        default=None, metadata={"help": "Path for redirecting Transformers logs to local file."}
    )

    def __post_init__(self):
        super().__post_init__()

        if self.resume_from_checkpoint is not None:
            if self.resume_from_checkpoint.lower() in ["false", 'f']:
                self.resume_from_checkpoint = None
            elif self.resume_from_checkpoint.lower() in ["true", 't']:
                self.resume_from_checkpoint = True
