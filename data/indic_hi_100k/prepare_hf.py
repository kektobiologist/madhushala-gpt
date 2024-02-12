
import os
import pickle
import requests
import numpy as np
from datasets import Dataset
from tqdm import tqdm

input_file_path = os.path.join(os.path.dirname(__file__), 'hi_100k_hf')
if not os.path.exists(input_file_path):
    print("couldn't find hi_100k_hf, download it first")
    exit(0)

# load the dataset
dataset = Dataset.load_from_disk(input_file_path)
# we'll go through dataset twice. first to create vocabulary,
# second to encode and write to bin files
chars = set()
for example in dataset:
    chars.update(set(example['text']))
vocab_size = len(chars)
# add an end of text token to vocabulary
candidate_eot = chr(0)
while candidate_eot in chars:
    candidate_eot = chr(ord(candidate_eot) + 1)
chars.add(candidate_eot)
eot = candidate_eot
chars = sorted(list(chars))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")
print(f"eot token (as number): {ord(eot):,} (as character): {eot}")
# check if \n is part of vocab
if '\n' in chars:
    print("newline character is part of the vocabulary")
# make a train/val split of 95/5
dataset = dataset.train_test_split(test_size=0.05, seed=2357, shuffle=True)
dataset['val'] = dataset.pop('test') # rename the test split to val

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

def process(example):
    ids = encode(example['text'])
    ids.append(stoi[eot]) # add the end of text token
    out = {'ids': ids, 'len': len(ids)}
    return out

num_proc = 8

# tokenize the dataset
tokenized = dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
    'eot': eot,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
