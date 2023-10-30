import sys
import os
import random
import logging
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

N = 8
path_prefix = sys.argv[1]

folder_name = "MINDlarge_train"     # Change to MINDsmall_train for tokenize MIND small

def wc_count(file_name):
    import subprocess
    out = subprocess.getoutput("wc -l %s" % file_name)
    return int(out.split()[0])

seed=2021
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

behaviors_fpath = f'{path_prefix}/{folder_name}/behaviors.tsv'
_total_lines = wc_count(behaviors_fpath)
logging.info(f"Read from {behaviors_fpath} with {_total_lines} lines.")

def get_sample(all_element, num_sample):
    if num_sample > len(all_element):
        return random.sample(all_element * (num_sample // len(all_element) + 1), num_sample)
    else:
        return random.sample(all_element, num_sample)

behaviors = []
with open(behaviors_fpath) as f:
    for _, line in enumerate(f):
        iid, uid, time, history, imp = line.strip().split('\t')
        impressions = [x.split('-') for x in imp.split(' ')]
        pos, neg = [], []
        for news_ID, label in impressions:
            if int(label) == 0:
                neg.append(news_ID)
            elif int(label) == 1:
                pos.append(news_ID)
        if len(pos) == 0:
            continue
        for pos_id in pos:
            neg_candidate = get_sample(neg, 4)
            neg_str = ' '.join(neg_candidate)
            new_line = '\t'.join([iid, uid, time, history, pos_id, neg_str]) + '\n'
            behaviors.append(new_line)

logging.info(len(behaviors))
random.shuffle(behaviors)

split_behaviors = [[] for _ in range(N)]
for i, line in enumerate(behaviors):
    split_behaviors[i % N].append(line)

for i in range(N):
    fpath = f'{path_prefix}/{folder_name}/behaviors_np{N}_{i}.tsv'
    logging.info(f'Writing {len(split_behaviors[i])} lines to {fpath}...')
    with open(fpath, 'w') as f:
        for line in split_behaviors[i]:
            f.write(line)
