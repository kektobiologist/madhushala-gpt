
import os
import pickle
import requests
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'Madhushala.txt')
if not os.path.exists(input_file_path):
    print("couldn't find Madhushala.txt, download it first")
    exit(0)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# copy the meta.pkl from data/indic_hi_100k to here
indic_meta_file_path = os.path.join(os.path.dirname(__file__), '..', 'indic_hi_100k', 'meta.pkl')
my_meta_file_path = os.path.join(os.path.dirname(__file__), 'meta.pkl')
if not os.path.exists(my_meta_file_path):
    # exec mv command
    os.system(f"cp {indic_meta_file_path} {my_meta_file_path}")

# read the pickle file
with open(my_meta_file_path, 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
stoi, itos = meta['stoi'], meta['itos']
eot = meta['eot']
# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
print("all the unique characters:", ''.join(chars))
print(f"vocab size (from indic): {vocab_size:,}")

# replace all '\n\n' in data iwth '\n' + eot
data = data.replace('\n\n', f"\n{eot}")

def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
