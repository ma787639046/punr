#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Dataset for Cot-MAE

@Author  :   Ma (Ma787639046@outlook.com)
'''
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import os
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForWholeWordMask

from arguments import DataTrainingArguments

from transformers.utils import logging
logger = logging.get_logger(__name__)

@dataclass
class PUNRCollator(DataCollatorForWholeWordMask):
    max_seq_length: int = 512
    mlm_probability: float = 0.15
    data_args: DataTrainingArguments = None

    def __post_init__(self):
        super().__post_init__()

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to it's ref.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512, mlm_probability=0.15, preprocessed_masked_lms: list=None, preprocessed_covered_indexes: set=None):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """

        cand_indexes = []
        for (i, token) in enumerate(input_tokens):
            if token in ["[CLS]", "[SEP]"]:
                continue

            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        random.shuffle(cand_indexes)
        num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * mlm_probability))))
        masked_lms = [] if not preprocessed_masked_lms else preprocessed_masked_lms
        covered_indexes = set() if not preprocessed_covered_indexes else preprocessed_covered_indexes
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)
                masked_lms.append(index)

        assert len(covered_indexes) == len(masked_lms)
        mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
        return mask_labels

    def _truncate(self, example: List[int], tgt_len: int=512):
        if len(example) <= tgt_len:
            return example
        trunc = len(example) - tgt_len
        trunc_left = random.randint(0, trunc)
        trunc_right = trunc - trunc_left

        truncated = example[trunc_left:]
        if trunc_right > 0:
            truncated = truncated[:-trunc_right]

        if not len(truncated) == tgt_len:
            print(len(example), len(truncated), trunc_left, trunc_right, tgt_len, flush=True)
            raise ValueError
        return truncated

    def _pad(self, seq, val=0, tgt_len: int=512):
        assert len(seq) <= tgt_len
        return seq + [val for _ in range(tgt_len - len(seq))]
    
    def encode_batch_examples(self, 
                              examples: List[Dict[str, List[int]]], 
                              mlm_prob: float=0.15,
                              num_special_tokens_to_add: int=2, 
                              preprocessed_text_and_mask: dict=None,   # For further mask
                              ):
        # Encode a batch of examples with Whole Word Mask
        encoded_examples = []
        masks = []
        mlm_masks = []

        # Preserve original tokens & mlm_masks for further mask
        preserved_original_tokens = []
        preserved_mlm_masks = []

        # Dynamic padding
        tgt_len = max([len(e['text']) for e in examples])
        tgt_len = min(tgt_len, self.max_seq_length)
        tgt_len_wo_special_tokens = tgt_len - num_special_tokens_to_add

        # WWM with further mask of 'anchor'
        for idx, e in enumerate(examples):
            if preprocessed_text_and_mask is not None:  # support of further mask
                e_trunc = preprocessed_text_and_mask['preserved_original_tokens'][idx][:tgt_len_wo_special_tokens]
                tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]
                preprocessed_mlm_mask = preprocessed_text_and_mask['preserved_mlm_masks'][idx][:tgt_len_wo_special_tokens]
                preprocessed_masked_lms = []
                preprocessed_covered_indexes = set()
                for _token_idx, _token_mask in enumerate(preprocessed_mlm_mask):
                    if _token_mask == 1:
                        preprocessed_masked_lms.append(_token_idx)
                        preprocessed_covered_indexes.add(_token_idx)
                mlm_mask = self._whole_word_mask(
                                tokens, 
                                mlm_probability=mlm_prob, 
                                preprocessed_masked_lms=preprocessed_masked_lms, preprocessed_covered_indexes=preprocessed_covered_indexes
                            )  # WWM
            else:
                e_trunc = self._truncate(e['text'], tgt_len=tgt_len_wo_special_tokens) # Truncate on both side, as implemented in BERT
                tokens = [self.tokenizer._convert_id_to_token(tid) for tid in e_trunc]
                mlm_mask = self._whole_word_mask(tokens, mlm_probability=mlm_prob)  # WWM
            preserved_original_tokens.append(e_trunc)
            preserved_mlm_masks.append(mlm_mask)
            mlm_mask = self._pad([0] + mlm_mask, tgt_len=tgt_len)
            mlm_masks.append(mlm_mask)

            encoded = self.tokenizer.encode_plus(e_trunc,
                add_special_tokens=True,
                max_length=tgt_len,
                padding="max_length",
                truncation=True,
                return_token_type_ids=False,
            )
            masks.append(encoded['attention_mask'])
            encoded_examples.append(encoded['input_ids'])

        inputs, labels = self.mask_tokens(
            torch.tensor(encoded_examples, dtype=torch.long),
            torch.tensor(mlm_masks, dtype=torch.long)
        )
        attention_mask = torch.tensor(masks)

        batch = {
            "input_ids": inputs,
            "labels": labels,
            "attention_mask": attention_mask,
            "input_ids_unmasked": torch.tensor(encoded_examples, dtype=torch.long),
        }

        preserved = {
            'preserved_original_tokens': preserved_original_tokens,
            'preserved_mlm_masks': preserved_mlm_masks,
        }

        return batch, preserved

    def mask_news_history(self, examples, mlm_prob=0.15):
        user_feature_batch, user_news_segment_batch, mlm_masks = [], [], []
        pre_spanmask_batch = []

        for click_docs in examples:
            # Concat User History
            click_docs_input_ids: List[List[int]] = [item['text'] for item in click_docs]

            # For masking whole spans
            mask_span_idxs = []
            if self.data_args.mask_whole_spans:
                ## Cal num of mask pos
                num_tokens = [len(item) for item in click_docs_input_ids]   # length of each spans
                idx_pool = list(range(len(num_tokens)))  # Choose a mask span from idx_pool

                max_predictions = 512   # make sure do not exceed 512
                num_to_predict = min(max_predictions, max(1, int(round(sum(num_tokens) * mlm_prob))))   # num of mask pos
                num_seg_tokens_to_predict = int(round(num_to_predict * self.data_args.mask_whole_spans_token_ratio))     # `mask_whole_spans_token_ratio` for masking

                while len(idx_pool) > 0 and num_seg_tokens_to_predict > 0:
                    idx_pool = list(filter(lambda x: num_tokens[x] <= num_seg_tokens_to_predict, idx_pool))
                    if len(idx_pool) > 0:
                        random.shuffle(idx_pool)
                        sampled_idx = idx_pool.pop(0)
                        mask_span_idxs.append(sampled_idx)
                        num_seg_tokens_to_predict -= num_tokens[sampled_idx]
            
            # Concat clicked docs
            concat_user_input_ids = []
            concat_user_input_ids.extend(click_docs_input_ids[0])
            news_segment_ids: List[int] = [0] * (len(click_docs_input_ids[0]) + 2)  # [CLS] + len(input_ids) + [SEP]

            # `pre_spanmask` holds the whole span mask
            pre_spanmask = [0]  # do not mask [CLS]
            if mask_span_idxs and (0 in mask_span_idxs):
                pre_spanmask.extend([1]*len(click_docs_input_ids[0]))   # mask span[0]
            else:
                pre_spanmask.extend([0]*len(click_docs_input_ids[0]))   # do not mask span[0]
            pre_spanmask.append(0)  # do not mask [SEP]

            if len(click_docs_input_ids) > 1:
                for _idx, _ids in enumerate(click_docs_input_ids[1:], start=1):
                    if len(concat_user_input_ids) >= self.max_seq_length:
                        break
                    concat_user_input_ids.append(self.tokenizer.sep_token_id)
                    concat_user_input_ids.extend(_ids)
                    news_segment_ids.extend([_idx] * (len(_ids) + 1))
                    # `pre_spanmask` holds the whole span mask
                    if mask_span_idxs and (_idx in mask_span_idxs):
                        pre_spanmask.extend([1]*len(_ids))   # mask span[_idx]
                    else:
                        pre_spanmask.extend([0]*len(_ids))   # do not mask span[_idx]
                    pre_spanmask.append(0)  # do not mask [SEP]
            
            # Tokenize concated impression input ids
            concat_user_input_ids = self.tokenizer.prepare_for_model(concat_user_input_ids,
                                                                    max_length=self.max_seq_length, 
                                                                    padding=False,  # do not padding
                                                                    truncation=True,
                                                                    return_token_type_ids=False,
                                                                    return_attention_mask=False,
                                                                    add_special_tokens=True,    # Add [CLS] & end [SEP]
                                                                    )

            # Truncate News Segment ids if needed
            news_segment_ids = {'input_ids': news_segment_ids[:len(concat_user_input_ids['input_ids'])]}

            # Truncate `pre_spanmask` if needed
            pre_spanmask = pre_spanmask[:len(concat_user_input_ids['input_ids'])]

            user_feature_batch.append(concat_user_input_ids)
            user_news_segment_batch.append(news_segment_ids)
            pre_spanmask_batch.append(pre_spanmask)
        
        # Pad as BatchEncoding class
        # For User Encoder, pad concated clicked docs
        user_feature_batch = self.tokenizer.pad(user_feature_batch,
                                                padding='longest',
                                                max_length=self.max_seq_length,
                                                return_attention_mask=True,
                                                return_tensors="pt",
                                            )
        
        # WWM
        mlm_masks = torch.zeros_like(user_feature_batch['input_ids'])
        for i, ids in enumerate(user_feature_batch['input_ids']):
            # Pre-mask whole span
            preprocessed_masked_lms = []
            preprocessed_covered_indexes = set()
            for _token_idx, _token_mask in enumerate(pre_spanmask_batch[i]):
                if _token_mask == 1:
                    preprocessed_masked_lms.append(_token_idx)
                    preprocessed_covered_indexes.add(_token_idx)
            # Mask remaining randomly
            tokens = [self.tokenizer._convert_id_to_token(int(tid)) for tid in ids if tid != self.tokenizer.pad_token_id]
            mlm_mask = self._whole_word_mask(
                tokens, 
                mlm_probability=mlm_prob, 
                preprocessed_masked_lms=preprocessed_masked_lms, 
                preprocessed_covered_indexes=preprocessed_covered_indexes
            )  # WWM + Pre-mask whole span
            for k, v in enumerate(mlm_mask):
                mlm_masks[i][k] = v
        
        unmasked_input_ids = user_feature_batch['input_ids'].clone()
        masked_inputs_ids, labels = self.mask_tokens(
            user_feature_batch['input_ids'],
            mlm_masks
        )

        # Pad News Segment ids   
        user_news_segment_ids = self.tokenizer.pad(user_news_segment_batch,
                                                padding='longest',
                                                max_length=self.max_seq_length,
                                                return_attention_mask=False,
                                                return_tensors="pt",
                                            )['input_ids']

        batch = {
            'input_ids': masked_inputs_ids,
            'attention_mask': user_feature_batch['attention_mask'],
            'news_segment_ids': user_news_segment_ids,
            'labels': labels,
            'input_ids_unmasked': unmasked_input_ids,
        }

        return batch
    
    def process_news_collection(self, examples):
        batch = self.mask_news_history(examples, mlm_prob=self.mlm_probability)
        # AE-Dec
        ae_dec_head_batch = self.mask_news_history(examples, mlm_prob=self.data_args.enc_head_mask_ratio)
        batch['ae_dec_head_input_ids'] = ae_dec_head_batch['input_ids']
        batch['ae_dec_head_labels'] = ae_dec_head_batch['labels']
        batch['ae_dec_head_attention_mask'] = ae_dec_head_batch['attention_mask']
        batch['ae_dec_head_input_ids_unmasked'] = ae_dec_head_batch['input_ids_unmasked']
        batch['ae_dec_head_news_segment_ids'] = ae_dec_head_batch['news_segment_ids']
        # AR-Dec
        batch['ar_dec_head_input_ids'] = batch['input_ids_unmasked'].clone()
        batch['ar_dec_head_labels'] = batch['input_ids_unmasked'].clone().masked_fill_(~(batch['attention_mask'].bool()), -100)  # [bs, seq_len]
        batch['ar_dec_head_attention_mask'] = batch['attention_mask'].clone()
        batch['ar_dec_head_news_segment_ids'] = batch['news_segment_ids'].clone()

        return batch

    def process_pretrain_corpus(self, examples):
        batch, enc_preserved_text_and_mask = self.encode_batch_examples(examples=examples['anchor'], mlm_prob=self.mlm_probability)
        batch['news_segment_ids'] = torch.zeros_like(batch['input_ids'])

        # AE-Dec (further mask)
        ae_dec_head_batch, _ = self.encode_batch_examples(examples=examples['anchor'], mlm_prob=self.data_args.enc_head_mask_ratio, preprocessed_text_and_mask=enc_preserved_text_and_mask)
        batch['ae_dec_head_input_ids'] = ae_dec_head_batch['input_ids']
        batch['ae_dec_head_labels'] = ae_dec_head_batch['labels']
        batch['ae_dec_head_attention_mask'] = ae_dec_head_batch['attention_mask']
        batch['ae_dec_head_input_ids_unmasked'] = ae_dec_head_batch['input_ids_unmasked']
        batch['ae_dec_head_news_segment_ids'] = torch.zeros_like(batch['ae_dec_head_input_ids'])

        # AR-Dec
        ## Here we fill the labels (or input ids of AR-Dec) [PAD] area with -100, to avoid loss calculatation
        ## on [PAD] area by CrossEntropy loss function
        ## Ignore klloss on [PAD]:
        ## [cls] xxx xxx xxx [SEP] [PAD] [PAD]
        ##   0    0   0   0    0   -100  -100
        ar_dec_head_batch = self.tokenizer.batch_encode_plus(
                                batch_text_or_text_pairs=[self._truncate(e['text'], tgt_len=self.max_seq_length - 2) for e in examples['anchor']],
                                add_special_tokens=True,
                                max_length=self.max_seq_length,
                                padding='longest',
                                truncation=True,
                                return_token_type_ids=False,
                                is_split_into_words=True,
                                return_tensors='pt',
                            )
        batch['ar_dec_head_input_ids'] = ar_dec_head_batch['input_ids']
        batch['ar_dec_head_labels'] = ar_dec_head_batch['input_ids'].clone().masked_fill_(~(ar_dec_head_batch['attention_mask'].bool()), -100)  # [bs, seq_len]
        batch['ar_dec_head_attention_mask'] = ar_dec_head_batch['attention_mask']
        batch['ar_dec_head_news_segment_ids'] = torch.zeros_like(batch['ar_dec_head_input_ids'])

        return batch

    def __call__(self, examples):
        if self.data_args.data_type == 'title':
            return self.process_news_collection(examples)
        elif self.data_args.data_type.lower() in ['wiki', 'wikibook', 'anchor']:
            # Packing
            unpacked = {'anchor': []}
            for text_dict in examples:
            # Anchor Text
                unpacked['anchor'].append({'text': text_dict['anchor']})
            return self.process_pretrain_corpus(unpacked)

@dataclass
class PUNRCollatorForSpanSampling(PUNRCollator):
    def __call__(self, examples):
        # examples = sum(examples, [])

        return super(PUNRCollatorForSpanSampling, self).__call__(examples)

class PUNRDatasetForSpanSampling(Dataset):
    def __init__(self, dataset, data_args):
        self.dataset = dataset
        self.data_args = data_args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        # Here we use title texts of the User History, as concated input span
        spans = self.dataset[item]['spans']
        spans = spans[ :self.data_args.user_log_length]
        _rets = [{'text': _span['title']} for _span in spans]
        return _rets
