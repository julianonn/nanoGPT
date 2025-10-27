"""
Based on the prepare.py scripts from the other subdirs in the nanoGPT repo.
"""
import os
import pickle
import requests
import numpy as np
import random
import json
import re

# download emma from project gutenberg
# input_file_path = os.path.join(os.path.dirname(__file__), 'emma.txt')
input_file_path = os.path.join(os.path.dirname(__file__), 'emma_remap_vocab', 'emma100.txt')

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

subdirs = {
    100: os.path.join(os.path.dirname(__file__), 'emma_remap_vocab'),
    # 100: os.path.join(os.path.dirname(__file__), 'emma_100pct'),
    # 75: os.path.join(os.path.dirname(__file__), 'emma_75pct'),
    # 50: os.path.join(os.path.dirname(__file__), 'emma_50pct'),
    # 25: os.path.join(os.path.dirname(__file__), 'emma_25pct'),
}

for pct, subdir in subdirs.items():
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    tmp_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', data)
    if pct == 100:
        samp_data = data
    else:
        keep_index = sorted(
            random.sample(
                range(len(tmp_sentences)), int(len(tmp_sentences) * (pct / 100))
            )
        )
        samp_data = '\n'.join([tmp_sentences[i] for i in keep_index])
        # samp_data = '\n'.join(
        #     random.sample(tmp_lines, int(len(tmp_lines) * (pct / 100)))
        # )
    
    # get all the unique characters that occur in this text
    chars = sorted(list(set(samp_data)))
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    def encode(s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(l):
        return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # create the train and test splits
    n = len(samp_data)
    train_data = samp_data[:int(n*0.9)]
    val_data = samp_data[int(n*0.9):]

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(subdir, 'train.bin'))
    val_ids.tofile(os.path.join(subdir, 'val.bin'))

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(subdir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    vocab_meta = {
        'pct sampled': pct, 
        'vocab_size': vocab_size,
        'unique_chars': chars,
        'total_train_tokens': len(train_ids),
        'total_val_tokens': len(val_ids),
    }
    with open(os.path.join(subdir, 'vocab_meta.json'), 'w') as f:
        json.dump(vocab_meta, f, indent=4)

    with open(os.path.join(subdir, f'emma{pct}.txt'), 'w') as f:
        f.write(samp_data)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
