import os
import json
from pathlib import Path
import logging
import sys
import numpy as np
import glob
import random
from typing import List, Dict, Tuple, Any
from tqdm.auto import tqdm
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler # fp16 training
from contextlib import nullcontext

from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers import (
    DataCollatorWithPadding,
    BatchEncoding
)

from datasets import load_dataset

from parameters import parse_args
from data_handler.dataloader import TrainDataset, TestDataset, TrainCollator, TestCollator
from data_handler.evalsampler import DistributedEvalSampler
from data_handler.preprocess import read_news_bert, get_doc_input_bert
from modeling.model_bert import ModelBert, MODEL_CLASSES
from utils.utils import (
    set_seed, 
    check_args_environment, 
    get_checkpoint, 
    latest_checkpoint, 
    init_process, 
)
from utils.metrics import roc_auc_score, ndcg_score, mrr_score, acc, MetricsDict

# Remove all handlers associated with the root logger object.
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    # force=True,
)

def ddp_spawner(args, func):
    '''
    Distributed training
    '''
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    args = check_args_environment(args)

    logging.info('-----------start multiprocessing------------')
    mp.spawn(func,
             args=(args,),
             nprocs=args.world_size,
             join=True)

def train(local_rank, args):
    set_seed(args.random_seed)

    if args.load_ckpt_name is not None:
        ckpt_path = get_checkpoint(args.model_dir, args.load_ckpt_name)
    else:
        ckpt_path = latest_checkpoint(args.model_dir)

    local_rank, global_rank, world_size = init_process(local_rank=local_rank, args=args)

    # Get Filepaths
    data_files = glob.glob(os.path.join(args.train_data_dir, args.filename_pat))
    data_files = list(sorted(data_files))

    # Load Dataset
    dataset = load_dataset('csv', data_files={"train": data_files}, delimiter='\t', column_names=["iid", "uid", "time", "history", "pos_id", "neg_str"], split="train")

    logging.info("[{}] contains {} samples {} steps".format(
        global_rank, len(dataset), len(dataset) // args.batch_size))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=True)

    news, news_index, category_dict, subcategory_dict = read_news_bert(
        os.path.join(args.train_data_dir, 'news.tsv'), args, tokenizer=tokenizer, mode='train'
    )

    # The implementation of data loading is not optimized
    # Needs to be refactored like normal dataloader
    # SHOULD NOT load ALL data in numpy ndarray at once
    # (: But it's okay for a finetune process
    news_title, news_category, news_subcategory = get_doc_input_bert(
        news, news_index, category_dict, subcategory_dict, args)

    model = ModelBert.build(args)

    if args.use_pretrain_model:
        ckpt = torch.load(args.pretrain_model_path, map_location='cpu')
        pretrained_dict = ckpt["model_state_dict"]
        model_dict = model.state_dict()
        remain_key = list(model_dict.keys())
        pretrained_key = []
        for k, v in pretrained_dict.items():
            if not k.startswith('student'):
                continue
            key = k
            model_dict[key].copy_(v)
            pretrained_key.append(key)
            remain_key.remove(key)

        model.load_state_dict(model_dict)

        if global_rank == 0:
            logging.info(f"loaded pretrain model: {args.pretrain_model_path}")
            print(f'{len(pretrained_key)} loaded pretrained parameters:')
            for k in pretrained_key:
                print(f'\t{k}')
            print(f'{len(remain_key)} randomly initialized parameters:')
            for k in remain_key:
                print(f'\t{k}')

        del ckpt
        torch.cuda.empty_cache()

        if args.bert_trainable_layer != list(range(12)):
            for param in model.news_encoder.parameters():
                param.requires_grad = False

            for index, layer in enumerate(model.news_encoder.bert.encoder.layer):
                if index in args.bert_trainable_layer:
                    logging.info(f"finetune block {index}")
                    for param in layer.parameters():
                        param.requires_grad = True

        if args.enable_gpu:
            model = model.to(local_rank)

        pretrained_param = []
        rest_param = []
        for name, param in model.named_parameters():
            if name in pretrained_key:
                pretrained_param.append(param)
            else:
                rest_param.append(param)

        optimizer = torch.optim.AdamW([
            {'params': pretrained_param, 'lr': args.pretrain_lr},
            {'params': rest_param, 'lr': args.lr}], amsgrad=False)

    else:
        if args.bert_trainable_layer != list(range(12)):
            for param in model.news_encoder.encoder.layer.parameters():
                param.requires_grad = False

            for index, layer in enumerate(model.news_encoder.encoder.layer):
                if index in args.bert_trainable_layer:
                    logging.info(f"finetune block {index}")
                    for param in layer.parameters():
                        param.requires_grad = True

        if args.enable_gpu:
            model = model.to(local_rank)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, amsgrad=False)

    word_dict = None

    if args.load_ckpt_name is not None:
        ckpt_path = get_checkpoint(args.model_dir, args.load_ckpt_name)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Model loaded from {ckpt_path}")

    if global_rank == 0:
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
    
    model = DDP(model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                )
    
    dist.barrier()  # synchronizes all processes
    logging.info(f"Global rank: {global_rank}, Local rank: {local_rank}, World Size: {world_size}")

    # Dataloader
    train_dataset = TrainDataset(
        news_text=news_title,
        news_index=news_index,
        dataset=dataset,
        tokenizer=tokenizer,
        args=args,
    )
    sampler = DistributedSampler(dataset=train_dataset)
    collator = TrainCollator(
        tokenizer=tokenizer,
        max_length=512,     # Concated inputs
        return_tensors="pt",
        max_length_title=args.num_words_title,
    )
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
    )

    bs_total = int(world_size * args.batch_size * args.grad_accu_step)
    t_total = int(len(train_dataset) * int(args.epochs - args.start_epoch) // bs_total)
    if args.use_linear_lr_scheduler:
        t_warmup = int(t_total * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=t_warmup, num_training_steps=t_total
        )
        if local_rank == 0:
            logging.info(f"Using Linear Scheduler with Warmup. Total optimization steps: {t_total}, Warmup steps: {t_warmup}")
    else:
        scheduler = None
    
    # fp16 training
    if args.fp16:
        scaler = GradScaler()

    if local_rank == 0:
        logging.info(f"***** Running training *****")
        logging.info(f"  Num examples = {len(train_dataset)}")
        logging.info(f"  Num Epochs = {int(args.epochs - args.start_epoch)}")
        logging.info(f"  Instantaneous batch size per device = {args.batch_size}")
        logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {bs_total}")
        logging.info(f"  Gradient Accumulation steps = {args.grad_accu_step}")
        logging.info(f"  Total optimization steps = {t_total}")

    for ep in range(args.start_epoch, args.epochs):
        if dist.is_initialized():
            sampler.set_epoch(ep)
        
        loss = 0.0
        accuary = 0.0
        for cnt, (user_inputs, news_inputs, targets) in enumerate(dataloader):
            user_inputs = user_inputs.to(local_rank)
            news_inputs = news_inputs.to(local_rank)
            targets = targets.to(local_rank)

            if cnt > args.max_steps_per_epoch:
                break
            
            with autocast() if args.fp16 else nullcontext():
                bz_loss, y_hat = model(user_inputs, news_inputs, targets)
            
            bz_loss = bz_loss / args.grad_accu_step     # Grad accu
            
            loss += bz_loss.data.float()
            accuary += acc(targets, y_hat)
            
            if args.fp16:
                scaler.scale(bz_loss).backward()
            else:
                bz_loss.backward()
            
            if (cnt + 1) % args.grad_accu_step == 0:
                if args.fp16:
                    # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
                    # do gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                args.max_grad_norm)
                    # update
                    scaler.step(optimizer)
                    scaler.update()
                    model.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()
                
                if scheduler is not None:
                    scheduler.step()

            if cnt % args.log_steps == 0 and local_rank == 0:
                logging.info(
                    '[{}] Epoch: {} Step: {} Ed: {}, lr: {}, train_loss: {:.5f}, acc: {:.5f}'.format(
                        global_rank, ep, cnt, cnt * args.batch_size, 
                        optimizer.param_groups[0]['lr'], 
                        loss.data / cnt,
                        accuary / cnt))

        loss /= cnt
        print(ep + 1, loss)

        # save model last of epoch
        if global_rank == 0:
            ckpt_path = os.path.join(args.model_dir, f'epoch-{ep+1}.pt')
            torch.save(
                {
                    'model_state_dict': model.module.state_dict(),
                    'category_dict': category_dict,
                    'word_dict': word_dict,
                    'subcategory_dict': subcategory_dict
                }, ckpt_path)
            logging.info(f"Model saved to {ckpt_path}")

    dist.destroy_process_group()    # Clean up dist groups


def test(local_rank, args):
    set_seed(args.random_seed)

    local_rank, global_rank, world_size = init_process(local_rank=local_rank, args=args)

    if args.load_ckpt_name is not None:
        ckpt_path = get_checkpoint(args.model_dir, args.load_ckpt_name)
    else:
        ckpt_path = latest_checkpoint(args.model_dir)

    assert ckpt_path is not None, 'No ckpt found'
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    subcategory_dict = checkpoint['subcategory_dict']
    category_dict = checkpoint['category_dict']
    word_dict = checkpoint['word_dict']

    model = ModelBert.build(args)

    if args.enable_gpu:
        model.to(local_rank)

    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model loaded from {ckpt_path}")

    model.eval()
    torch.set_grad_enabled(False)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer: PreTrainedTokenizerBase = tokenizer_class.from_pretrained(args.tokenizer_name, do_lower_case=True)

    news, news_index = read_news_bert(
        os.path.join(args.test_data_dir, 'news.tsv'), args, tokenizer=tokenizer, mode='test'
    )
    
    news_title, news_category, news_subcategory = get_doc_input_bert(
        news, news_index, category_dict, subcategory_dict, args)
    
    # For News Encoder
    class NewsDataset(Dataset):
        def __init__(self, data, tokenizer, args):
            self.data = data
            self.tokenizer = tokenizer
            self.args = args

        def __getitem__(self, idx):
            news_feature = self.data[idx]
            news_feature = self.tokenizer.prepare_for_model(news_feature,
                                                            truncation=True,
                                                            padding=False,
                                                            max_length=self.args.num_words_title,
                                                            return_token_type_ids=False,
                                                            return_attention_mask=False,
                                                            return_tensors="pt",
                                                        )
            news_feature["news_segment_ids"] = torch.zeros_like(news_feature['input_ids'])
            return news_feature

        def __len__(self):
            return len(self.data)

    class NewsCollatorTest(DataCollatorWithPadding):
        def __init__(self, *args, **kwargs):
            return super().__init__(*args, **kwargs)
        
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            batch = self.tokenizer.pad([{"input_ids": item["input_ids"]} for item in features],
                                        padding=self.padding,
                                        max_length=self.max_length,
                                        pad_to_multiple_of=self.pad_to_multiple_of,
                                        return_tensors=self.return_tensors,
                                        )
            batch["news_segment_ids"] = self.tokenizer.pad([{"input_ids": item["news_segment_ids"]} for item in features],
                                                            padding=self.padding,
                                                            max_length=self.max_length,
                                                            pad_to_multiple_of=self.pad_to_multiple_of,
                                                            return_tensors=self.return_tensors,
                                                            )["input_ids"]
            return batch
    
    news_collate_fn = NewsCollatorTest(tokenizer=tokenizer, padding='max_length', max_length=args.num_words_title)
    news_dataset = NewsDataset(news_title, tokenizer, args)
    news_dataloader = DataLoader(news_dataset,
                                batch_size=args.eval_batch_size,
                                # num_workers=args.num_workers,
                                num_workers=0,
                                collate_fn=news_collate_fn)

    logging.info(f"Forwarding through News Encoder...")
    news_scoring = []
    with torch.no_grad():
        for news_inputs in tqdm(news_dataloader):
            news_inputs: BatchEncoding = news_inputs.to(torch.cuda.current_device())
            news_vec = model.get_news_vec(news_inputs).cpu().detach().numpy()
            news_scoring.extend(news_vec)

    news_scoring = np.array(news_scoring)
    logging.info("news scoring num: {}".format(news_scoring.shape[0]))

    # Get Filepaths
    data_files = glob.glob(os.path.join(args.test_data_dir, args.filename_pat))
    data_files = list(sorted(data_files))

    # Load Dataset
    dataset = load_dataset('csv', data_files={"test": data_files}, delimiter='\t', column_names=["iid", "uid", "time", "history", "impression"], split="test")

    # Dataloader
    eval_dataset = TestDataset(
        news_text=news_title,
        news_index=news_index,
        dataset=dataset,
        tokenizer=tokenizer,
        args=args,
    )
    sampler = DistributedEvalSampler(dataset=eval_dataset)
    collator = TestCollator(
        tokenizer=tokenizer,
        max_length=512,     # Concated inputs
        return_tensors="pt",
        max_length_title=args.num_words_title,
    )
    dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.eval_batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=args.num_workers,
    )
    local_sample_num = 0

    local_sample_num_cold = 0
    
    results = MetricsDict(metrics_name=["AUC", "MRR", "nDCG5", "nDCG10"])
    results.add_metric_dict('all users')
    results.add_metric_dict('cold users(k<=5)')
    results.add_metric_dict('cold users(k=0)')
    results.add_metric_dict('cold users(k=1)')
    results.add_metric_dict('cold users(k=3)')
    results.add_metric_dict('cold users(k=5)')

    #Synchronizes
    dist.barrier()
    torch.cuda.empty_cache()

    # Saving predict results
    pred_results = []

    for cnt, (user_feature_batch, news_idxs_batch, label_batch, history_length_batch, impression_ids_batch, user_ids_batch, click_docs_batch, candidate_sets_batch) in enumerate(dataloader):

        local_sample_num += user_feature_batch['input_ids'].shape[0]
        user_feature_batch: BatchEncoding = user_feature_batch.to(local_rank)
        
        # Forward to get User vector
        user_vecs = model.get_user_vec(user_feature_batch).cpu().detach().numpy()

        for user_vec, news_idxs, label, history_length, impression_id, user_id, click_docs, candidate_sets in zip(user_vecs, news_idxs_batch, label_batch, history_length_batch, impression_ids_batch, user_ids_batch, click_docs_batch, candidate_sets_batch):
            if label.mean() == 0 or label.mean() == 1:
                continue
            news_vec = news_scoring[news_idxs]
            score = np.dot(news_vec, user_vec)
            
            _rank = []
            _score = []
            sorted_indices = list(np.argsort(score)[::-1])
            for i in sorted_indices:
                _rank.append(candidate_sets[i])
                _score.append(float(score[i]))
            
            _label = [candidate_sets[idx] for idx in np.where(label == np.max(label))[0]]
            
            pred = {
                "impression_id": int(impression_id),
                "user_id": str(user_id),
                "click_docs": click_docs,
                "candidate_sets": candidate_sets,
                "predict": _rank,
                "score_of_predict": _score, 
                "label": _label,
                "hit_at": [_rank.index(_i) for _i in _label],
            }
            pred_results.append(pred)
            metric_rslt = results.cal_metrics(score, label)
            results.update_metric_dict('all users', metric_rslt)

            if history_length == 0:
                results.update_metric_dict('cold users(k=0)', metric_rslt)
            
            if history_length == 1:
                results.update_metric_dict('cold users(k=1)', metric_rslt)
            
            if history_length == 3:
                results.update_metric_dict('cold users(k=3)', metric_rslt)
            
            if history_length == 5:
                results.update_metric_dict('cold users(k=5)', metric_rslt)

            if history_length <= 5:
                results.update_metric_dict('cold users(k<=5)', metric_rslt)
                local_sample_num_cold += 1

        if cnt % args.log_steps == 0 and local_rank == 0:
            results.print_metrics(local_rank, cnt * args.batch_size, 'all users')

    logging.info('[{}] local_sample_num (cold user): {}'.format(global_rank, local_sample_num_cold))
    logging.info('[{}] local_sample_num (all user): {}'.format(global_rank, local_sample_num))

    dist.barrier()

    save_pred_path = Path(args.model_dir).parent / f"pred_results_rank{local_rank}.json"
    with open(save_pred_path, "w") as f:
        json.dump(pred_results, f, indent=2)

    if local_rank == 0:
        logging.info('Cold Users AllReduced Dev Results: ')
    results.print_reduced_metrics(local_rank, len(eval_dataset), 'cold users(k=0)')
    results.print_reduced_metrics(local_rank, len(eval_dataset), 'cold users(k=1)')
    results.print_reduced_metrics(local_rank, len(eval_dataset), 'cold users(k=3)')
    results.print_reduced_metrics(local_rank, len(eval_dataset), 'cold users(k=5)')
    results.print_reduced_metrics(local_rank, len(eval_dataset), 'cold users(k<=5)')

    if local_rank == 0:
        logging.info('All Users AllReduced Dev Results: ')
    results.print_reduced_metrics(local_rank, len(eval_dataset), 'all users')
    
    dist.destroy_process_group()    # Clean up dist groups


if __name__ == "__main__":
    args = parse_args()
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)

    if 'train' in args.mode:
        ddp_spawner(args, train)

    if 'test' in args.mode:
        ddp_spawner(args, test)
