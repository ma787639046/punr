import sys
import traceback
import logging
import random
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from dataclasses import dataclass
from torch.utils.data import IterableDataset, Dataset
from transformers.data import DataCollatorWithPadding
from typing import Any, List, Dict, Tuple, Union
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

class TrainDataset(Dataset):
    def __init__(self, 
                 news_text: List[List[int]],    # intid -> Tokenized input ids
                 news_index: Dict[str, int],    # doc_strid -> intid
                 dataset,   # Iterable hfdataset 
                 tokenizer, # tokenizer
                 args,      # args
                 ):
        self.news_text = news_text
        self.news_index = news_index
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.args = args

    def __len__(self):
        return len(self.dataset)
    
    def trans_to_nindex(self, nids, padding=True):
        if padding:     # Replace all unexist news ids with pad doc `0`
            return [self.news_index[i] if i in self.news_index else 0 for i in nids]
        else:           # Do not padding, except no exist news
            ret = []
            for i in nids:
                if i in self.news_index:
                    ret.append(self.news_index[i])
            if len(ret) == 0:
                ret = [0]   # Use pad doc
            return ret
    
    def concat_user_inputs(self, click_docs: List[str]):
        click_docs_input_ids, concat_user_input_ids = [], []
        # Concat clicked docs
        click_docs = click_docs[:self.args.user_log_length]
        # Fetch news input ids
        click_docs_input_ids: List[List[int]] = [self.news_text[i] for i in self.trans_to_nindex(click_docs, padding=False)]
        # Start to concat
        concat_user_input_ids.extend(click_docs_input_ids[0])
        news_segment_ids: List[int] = [0] * (len(click_docs_input_ids[0]) + 2)  # [CLS] + len(input_ids) + [SEP]
        if len(click_docs_input_ids) > 1:
            for _idx, _ids in enumerate(click_docs_input_ids[1:], start=1):
                concat_user_input_ids.append(self.tokenizer.sep_token_id)
                concat_user_input_ids.extend(_ids)
                news_segment_ids.extend([_idx] * (len(_ids) + 1))
        
        # Tokenize concated impression input ids
        user_max_length = 512
        concat_user_input_ids = self.tokenizer.prepare_for_model(concat_user_input_ids,
                                                                max_length=user_max_length, 
                                                                padding=False,  # do not padding
                                                                truncation=True,
                                                                return_token_type_ids=False,
                                                                return_attention_mask=False,
                                                                add_special_tokens=True,
                                                                )
        # Truncate News Segment ids if needed
        news_segment_ids = {'input_ids': news_segment_ids[:len(concat_user_input_ids['input_ids'])]}

        return concat_user_input_ids, news_segment_ids

    def __getitem__(self, idx):
        line = self.dataset[idx]
        # line = line.decode(encoding="utf-8").split("\t")    # split
        # click_docs = line[3].split()    # Clicked history docid
        # sess_pos = line[4].split()      # Positive docid
        # sess_neg = line[5].split()      # Negative docid

        click_docs = line['history'].split() if line['history'] else []   # Clicked history docid
        sess_pos = line['pos_id'].split()      # Positive docid
        sess_neg = line['neg_str'].split()      # Negative docid

        # Concat User inputs
        concat_user_input_ids, news_segment_ids = self.concat_user_inputs(click_docs)

        # Construct Candidate News
        pos = self.trans_to_nindex(sess_pos)
        neg = self.trans_to_nindex(sess_neg)

        label = random.randint(0, self.args.npratio)
        sample_news = neg[:label] + pos + neg[label:]

        news_feature_input_ids = []
        for i in sample_news:
            news_feature = self.tokenizer.prepare_for_model(self.news_text[i],
                                                            max_length=self.args.num_words_title, 
                                                            padding=False,  # do not padding
                                                            truncation=True,
                                                            return_token_type_ids=False,
                                                            return_attention_mask=False,
                                                            add_special_tokens=True,
                                                            )
            news_feature_input_ids.append(news_feature)

        rets = {
            # User inputs
            "user_input_ids": concat_user_input_ids,            # List[int]
            "user_news_segment_ids": news_segment_ids,        # List[int]
            # News Train
            "news_feature_input_ids": news_feature_input_ids,   # List[List[int]]
            "label": label,                                     # List[int]
        }
        
        return rets

class TestDataset(TrainDataset):

    def __getitem__(self, idx): # ["iid", "uid", "time", "history", "pos_id", "neg_str"]
        line = self.dataset[idx]
        # line = line.decode(encoding="utf-8").split("\t")    # split
        # click_docs = line[3].split()    # Clicked history docid
        click_docs = line["history"].split() if line['history'] else []    # Clicked history docid
        impression = line["impression"].split()

        # Concat User inputs
        concat_user_input_ids, news_segment_ids = self.concat_user_inputs(click_docs)
        
        # For News sample
        news_idxs = self.trans_to_nindex([i.split('-')[0] for i in impression])
        label = np.array([int(i.split('-')[1]) for i in impression])

        rets = {
            # User inputs
            "user_input_ids": concat_user_input_ids,            # List[int]
            "user_news_segment_ids": news_segment_ids,        # List[int]
            # News Test
            "news_idxs": news_idxs,         # List[int]
            "test_label": label,            # List[int]
            "history_length": len(click_docs),   # int
            "iid": line["iid"],
            "uid": line["uid"],
            "click_docs": click_docs,
            "candidate_sets": [i.split('-')[0] for i in impression],
        }

        return rets


@dataclass
class TrainCollator(DataCollatorWithPadding):
    max_length_title: int = None

    def pad_user_inputs(self, user_feature_batch, user_news_segment_batch):
        # Pad as BatchEncoding class
        # For User Encoder, pad concated clicked docs
        user_feature_batch = self.tokenizer.pad(user_feature_batch,
                                                padding='longest',
                                                max_length=self.max_length,
                                                return_attention_mask=True,
                                                return_tensors=self.return_tensors,
                                            )
        user_news_segment_batch = self.tokenizer.pad(user_news_segment_batch,
                                                padding='longest',
                                                max_length=self.max_length,
                                                return_attention_mask=False,
                                                return_tensors=self.return_tensors,
                                            )
        user_feature_batch["news_segment_ids"] = user_news_segment_batch['input_ids']
        return user_feature_batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Unpacking
        user_feature_batch = [item["user_input_ids"] for item in features]
        user_news_segment_batch = [item["user_news_segment_ids"] for item in features]
        news_feature_batch = [item["news_feature_input_ids"] for item in features]
        news_feature_batch = sum(news_feature_batch, [])
        label_batch = [item["label"] for item in features]

        # Pad User inputs
        user_feature_batch = self.pad_user_inputs(user_feature_batch, user_news_segment_batch)

        # For News Encoder, padding as one news segment
        news_feature_batch = self.tokenizer.pad(news_feature_batch,
                                                padding='longest',
                                                max_length=self.max_length_title,
                                                return_attention_mask=True,
                                                return_tensors=self.return_tensors,
                                            )
        news_feature_batch["news_segment_ids"] = torch.zeros_like(news_feature_batch['input_ids'])

        # Label
        label_batch = torch.LongTensor(label_batch)

        return user_feature_batch, news_feature_batch, label_batch

@dataclass
class TestCollator(TrainCollator):
    def __call__(self, features: List[Dict[str, Any]]):
        # Unpacking
        user_feature_batch = [item["user_input_ids"] for item in features]
        user_news_segment_batch = [item["user_news_segment_ids"] for item in features]
        news_idxs_batch = [item["news_idxs"] for item in features]
        label_batch = [item["test_label"] for item in features]
        history_length_batch = [item["history_length"] for item in features]

        impression_ids_batch = [item["iid"] for item in features]
        user_ids_batch = [item["uid"] for item in features]
        click_docs_batch = [item["click_docs"] for item in features]
        candidate_sets_batch = [item["candidate_sets"] for item in features]

        # Pad User inputs
        user_feature_batch = self.pad_user_inputs(user_feature_batch, user_news_segment_batch)

        return user_feature_batch, news_idxs_batch, label_batch, history_length_batch, impression_ids_batch, user_ids_batch, click_docs_batch, candidate_sets_batch

