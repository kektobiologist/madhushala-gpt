import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

data_dir = os.path.dirname(__file__)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

block_size = 128
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    # convert x to string using the meta information
    if meta_vocab_size is not None:
        x_np = x.numpy()
        x_str = ''.join([meta['itos'][i] for i in x_np[0]])
        print(f"x_str: {x_str}")
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

get_batch('train')
get_batch('val')

# test the encode/decode functions from the meta information
if meta_vocab_size is not None:
    print("testing encode/decode functions")
    s = "hello, world!\n"
    print(f"original string: {s}")
    l = [meta['stoi'][c] for c in s]
    print(f"encoded list: {l}")
    s2 = ''.join([meta['itos'][i] for i in l])
    print(f"decoded string: {s2}")