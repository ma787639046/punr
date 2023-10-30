#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Tokenize MIND corpus
This data holds spans of same document, to easily add contrastive learning objects if needed.

@Author  :   Ma (Ma787639046@outlook.com)
'''

import os
import random
import argparse
from typing import List
from math import floor
import json
from transformers import AutoTokenizer
from multiprocessing import Pool
from tqdm import tqdm
from collections import defaultdict

def wc_count(file_name):
    import subprocess
    out = subprocess.getoutput("wc -l %s" % file_name)
    return int(out.split()[0])

seed=42
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--files',
    default=[
        "data/MINDlarge_train/news.tsv",
    ],
    nargs="+",
    help="Path to tsv data file."
)
parser.add_argument(
    '--behavior',
    default="data/MINDlarge_train/behaviors.tsv",
)
parser.add_argument(
    '--save_to',
    default="data/pretrain/mindlarge_user_title.multispans.json",
)
parser.add_argument(
    '--maxlen',
    default=50,
    type=int,
    required=False
)
parser.add_argument(
    '--tokenizer',
    default="bert-base-uncased",
    required=False
)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

def encode_one_line(line: str, maxlen=args.maxlen) -> List[List[List[int]]]:
    doc_id, category, subcategory, title, abstract, url, title_entities, abstract_entities = line.strip().split('\t')
    tokenized_title = tokenizer(title,
                          add_special_tokens=False,
                          truncation=False,
                          return_attention_mask=False,
                          return_token_type_ids=False,
                        )["input_ids"]
    
    if len(tokenized_title) == 0:
        return None
    
    _span = {
        'title': tokenized_title,
    }

    return _span

# Mkdir if not exists
if '/' in args.save_to:
    dir_path = os.path.split(args.save_to)[0]
    if dir_path is not None and len(dir_path) > 0:
        os.makedirs(dir_path, exist_ok=True)

# Read news.tsv from multiple files
news_collections = defaultdict(str)
for _file in args.files:
    with open(_file, 'r') as f:
        for line in tqdm(f, total=wc_count(_file)):
            doc_id, category, subcategory, title, abstract, url, title_entities, abstract_entities = line.strip().split('\t')
            news_collections[doc_id] = encode_one_line(line)

print(f"Total {len(news_collections)} news in collections.")

def get_history(line: str):
    iid, uid, time, history, imp = line.strip().split('\t')
    history_list = history.split(' ')
    ret = []
    for i in history_list:
        ret.append(news_collections[i.strip()])
        
    if len(ret) <= 1:
        return None

    return json.dumps({'spans': ret})

with open(args.save_to, 'w') as f:
    # Multiprocess is highly recommended
    with Pool(5) as p:
        all_tokenized = p.imap_unordered(
            get_history,
            tqdm(open(args.behavior), total=wc_count(args.behavior)),
            chunksize=1000,
        )
        for _span in all_tokenized:
            if not _span:
                continue
            f.write(_span + '\n')

