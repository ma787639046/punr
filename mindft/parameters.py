import argparse
from utils import utils
import logging

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        type=str,
                        default="train_test",
                        choices=['train', 'test', 'train_test'])
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        required=True,
    )
    parser.add_argument("--filename_pat", type=str, default="behaviors_*.tsv")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        required=True,
        help='Where to save the finetuned model.'
    )
    parser.add_argument("--npratio", type=int, default=4)
    parser.add_argument("--enable_shuffle", type=utils.str2bool, default=True)
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--filter_num", type=int, default=3)
    parser.add_argument("--log_steps", type=int, default=20)
    parser.add_argument("--random_seed", type=int, default=42)

    # DDP Training
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=int, default=2234)
    parser.add_argument("--enable_gpu", type=utils.str2bool, default=True)
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--n_gpu", type=int, default=8)     # 4 GPUs, batch_size 32; 8 GPUs, batch_size 16
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=300)
    parser.add_argument("--fp16", type=utils.str2bool, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_accu_step", type=int, default=1)
    parser.add_argument("--use_linear_lr_scheduler", type=utils.str2bool, default=True)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # model training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)

    parser.add_argument("--num_words_title", type=int, default=30)
    parser.add_argument("--num_words_abstract", type=int, default=50)
    parser.add_argument("--num_words_body", type=int, default=100)
    parser.add_argument(
        "--user_log_length",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--word_embedding_dim",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--glove_embedding_path",
        type=str,
        default='./glove.840B.300d.txt',
    )
    parser.add_argument("--freeze_embedding",
                        type=utils.str2bool,
                        default=False)  # Not working (TODO: Fix)
    parser.add_argument(
        "--news_dim",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--news_query_vector_dim",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--user_query_vector_dim",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--num_attention_heads",
        type=int,
        default=12,
        help="Control Nums of Attn Head when uses NRMS model in UserEncoder.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Control Hidden Dim of each Attn Head when uses NRMS model in UserEncoder.",
    )
    parser.add_argument(
        "--user_log_mask", 
        type=utils.str2bool, 
        default=False,
        help="News vectors pad doc for UserEncoder."
    )
    parser.add_argument("--drop_rate", type=float, default=0.2)
    parser.add_argument("--save_steps", type=int, default=50000)
    parser.add_argument("--max_steps_per_epoch", type=int, default=1000000)

    parser.add_argument(
        "--load_ckpt_name",
        type=str,
        default=None,
        help="choose which ckpt to load and test"
    )

    # bert
    parser.add_argument("--apply_bert", type=utils.str2bool, default=True)  # Not working
    parser.add_argument("--model_type", default="bert", type=str)
    parser.add_argument("--do_lower_case", type=utils.str2bool, default=True)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--config_name", required=True, type=str)
    parser.add_argument("--tokenizer_name", required=True, type=str)

    parser.add_argument("--num_hidden_layers", type=int, default=12)
    parser.add_argument(
        "--bert_trainable_layer",
        type=int, nargs='+',
        default=list(range(12)),
        choices=list(range(12)))

    parser.add_argument(
        "--model", 
        type=str, 
        default='NRMS',
        help="NRMS: Use Multi-head self-attn over history news emb of NewsEncoder in UserEncoder.",
    )
    parser.add_argument(
        "--pooling", 
        type=str, 
        default='cls',
        help="att; cls; avg."
             "Pooling for all NewsEncoder. [bs, seq_len, hid] -> [bs, hid]",
    )
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--use_pretrain_model", type=utils.str2bool, default=False)
    parser.add_argument("--pretrain_model_path", type=str, default=None)
    parser.add_argument("--pretrain_lr", type=float, default=0.00005)
    parser.add_argument(
        "--untied_encoder", 
        type=utils.str2bool,
        default=False,
        help="set to True for seperated encoder."
    )

    args = parser.parse_args()

    logging.info(args)
    return args


if __name__ == "__main__":
    args = parse_args()
