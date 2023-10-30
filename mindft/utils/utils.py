import logging
import os
import sys
import torch
import numpy as np
import argparse
import re
import torch.distributed as dist
import random

def word_tokenize(sent):
    pat = re.compile(r'[\w]+|[.,!?;|]')
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def check_args_environment(args):
    if not torch.cuda.is_available():
        logging.warning("Cuda is not available, " \
                        "related options will be disabled")
    args.enable_gpu = torch.cuda.is_available() & args.enable_gpu
    return args

def init_process(local_rank: int, args):
    global_rank: int = args.node_rank * args.n_gpu + local_rank
    world_size: int = args.world_size

    # initialize the process group
    os.environ['MASTER_ADDR'] = str(args.master_addr)
    os.environ['MASTER_PORT'] = str(args.master_port)
    os.environ["RANK"] = str(local_rank)
    dist.init_process_group(backend='nccl', rank=local_rank, world_size=world_size, )
    torch.cuda.set_device(local_rank)

    return local_rank, global_rank, world_size

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def dump_args(args):
    for arg in dir(args):
        if not arg.startswith("_"):
            logging.info(f"args[{arg}]={getattr(args, arg)}")

def load_matrix(embedding_file_path, word_dict, word_embedding_dim):
    embedding_matrix = np.zeros(shape=(len(word_dict) + 1,
                                               word_embedding_dim))
    have_word = []
    if embedding_file_path is not None:
        with open(embedding_file_path, 'rb') as f:
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                line = line.split()
                word = line[0].decode()
                if word in word_dict:
                    index = word_dict[word]
                    tp = [float(x) for x in line[1:]]
                    embedding_matrix[index] = np.array(tp)
                    have_word.append(word)
    return embedding_matrix, have_word

def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    all_checkpoints = {
        int(x.split('.')[-2].split('-')[-1]): x
        for x in os.listdir(directory)
    }
    if not all_checkpoints:
        return None
    return os.path.join(directory,
                        all_checkpoints[max(all_checkpoints.keys())])

def get_checkpoint(directory, ckpt_name):
    ckpt_path = os.path.join(directory, ckpt_name)
    if os.path.exists(ckpt_path):
        return ckpt_path
    else:
        return None


@torch.no_grad()
def varsize_gather_nograd(x: torch.Tensor):
    """gather tensors of different sizes along the first dimension"""
    if not dist.is_initialized():
        return x

    # determine max size
    size = torch.tensor([x.shape[0]], device=x.device, dtype=torch.int)
    allsizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(allsizes, size)
    max_size = max([size.cpu().max() for size in allsizes])

    padded = torch.empty(max_size, *x.shape[1:], dtype=x.dtype, device=x.device)
    padded[: x.shape[0]] = x
    output = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(output, padded)

    output = [tensor[: allsizes[k]] for k, tensor in enumerate(output)]
    output = torch.cat(output, dim=0)

    return output


@torch.no_grad()
def get_varsize(x: torch.Tensor):
    """gather tensors of different sizes along the first dimension"""
    if not dist.is_initialized():
        return [x.shape[0]]

    # determine max size
    size = torch.tensor([x.shape[0]], device=x.device, dtype=torch.int)
    allsizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
    dist.all_gather(allsizes, size)
    allsizes = torch.cat(allsizes)
    return 

def weighted_average(x, count):
    if not dist.is_initialized():
        if isinstance(x, torch.Tensor):
            x = x.item()
        return x, count
    t_loss = torch.tensor([x * count]).cuda()
    t_total = torch.tensor([count]).cuda()
    t_loss = sum_main(t_loss)
    t_total = sum_main(t_total)
    return (t_loss / t_total).item(), t_total.item()
