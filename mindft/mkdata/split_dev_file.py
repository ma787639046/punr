import sys
import random
from tqdm import tqdm

N = 8
path_prefix = sys.argv[1]

folder_name = "MINDlarge_dev"     # Change to MINDsmall_dev for tokenize MIND small

def get_sample(all_element, num_sample):
    if num_sample > len(all_element):
        return random.sample(all_element * (num_sample // len(all_element) + 1), num_sample)
    else:
        return random.sample(all_element, num_sample)

behaviors = []
with open(f'{path_prefix}/{folder_name}/behaviors.tsv') as f:
    for line in tqdm(f):
        behaviors.append(line)

print(len(behaviors))

split_behaviors = [[] for _ in range(N)]
for i, line in enumerate(behaviors):
    split_behaviors[i % N].append(line)

for i in range(N):
    with open(f'{path_prefix}/{folder_name}/behaviors_{i}.tsv', 'w') as f:
        for line in split_behaviors[i]:
            f.write(line)
