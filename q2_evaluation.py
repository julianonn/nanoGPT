from multiprocessing.util import debug
import torch
import subprocess
import sys
from model import GPT, GPTConfig
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
import time
import numpy as np
import os
from tqdm import tqdm
import re
import math

SHAKESPEARE_CHAR_DIRS = [
        'out-shakespeare-char-mps-2L-2H',
        'out-shakespeare-char-mps-2L-4H',
        'out-shakespeare-char-mps-4L-2H',
        'out-shakespeare-char-mps-4L-4H',
        'out-shakespeare-char-mps-4L-8H',
        'out-shakespeare-char-mps-8L-4H',
        'out-shakespeare-char-mps-8L-8H',
    ]


######################################################
########     SPECIFIC EVALUATION METRIC:      ########
########            KL-DIVERGENCE             ########
######################################################

def kl_eval(input_data=os.path.join('shakespeare_char', 'input.txt'),
            model_directories=SHAKESPEARE_CHAR_DIRS, 
            ngram=3, 
            device='mps',
            write=True, 
            output_json='kl_eval.json'):
    
    print("----- KL DIVERGENCE -----")
    # read input file
    with open(os.path.join('data', input_data), 'r') as f:
        input_text = f.read()
    print('creating input ngram dist')
    input_dist = ngram_dist(input_text, n=ngram)
    res = []
    for d in model_directories:
        try:
            print(f'sampling from model in {d}')
            samp = sample(d, device=device, debug=False)
            samp_dist = ngram_dist(samp, n=ngram)
            print('computing kl divergence')
            kl_value = kl_div(input_dist, samp_dist).item()
            res.append({
                'model_dir': d,
                'kl_div_loss': kl_value,
            })
        except Exception as e:
            print(f"error, skipping model")
    print('writing results to', output_json)
    if write:
        with open(output_json, 'w') as f:
            tmp = {'time': time.time(), 'results': res}
            json.dump(tmp, f, indent=4)
    return res

def kl_div(input_dist: Counter, target_dist: Counter):
    ngrams = set(input_dist.keys()).union(set(target_dist.keys()))
    ngrams = sorted(ngrams)
    input_tensor = torch.tensor([input_dist.get(k, 0) for k in ngrams], dtype=torch.float32)
    target_tensor = torch.tensor([target_dist.get(k, 0) for k in ngrams], dtype=torch.float32)

    # following https://docs.pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    input_log = F.log_softmax(input_tensor, dim=-1)
    target_log = F.log_softmax(target_tensor, dim=-1)
    return kl_loss(input_log, target_log)


def sample(out_dir, device='mps', debug=True):
    cmd = [sys.executable, '-u', 'sample.py', f'--out_dir={out_dir}', f'--device={device}', '--compile=False']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    
    lines = []
    for line in iter(process.stdout.readline, ''):
        if debug:
            print(line, end='', flush=True) 
        lines.append(line)
    
    process.wait()
    samp = ''.join(lines)
    samp = samp.split("meta.pkl...")[1]
    samp = samp.replace("---------------", "")
    if debug:
        with open('temp.txt', 'w') as f:
            f.write(samp)
    return samp

def ngram_dist(txt, n=3):
    # character ngrams from text, 3gram by default
    ngrams = [txt[i:i+n] for i in range(len(txt)-(n-1))]
    dist = Counter(ngrams)
    return dist



######################################################
########     GENERAL EVALUATION METRIC:      ########
########             PERPLEXITY              ########
######################################################

def ppl_eval(model_directories=SHAKESPEARE_CHAR_DIRS,
             config='eval_shakespeare_char_mps.py',
             write=True,
             output_json='ppl_eval.json'):
    print("----- PERPLEXITY -----")
    res = []
    for d in model_directories:
        try:
            print(f'\n+++ evaluating model in {d} +++')
            dct = {'model_dir': d}
            losses = get_losses(d, config=config)
            dct.update(losses)
            dct['train_ppl'] = math.exp(losses['train_loss'])
            dct['val_ppl'] = math.exp(losses['val_loss'])
            res.append(dct)
        except Exception as e:
            print(f"error, skipping model")
    
    if write:
        with open(output_json, 'w') as f:
            tmp = {'time': time.time(), 'results': res}
            json.dump(tmp, f, indent=4)
    return res


def get_losses(current_model,
               config):
    cmd = ['python', 'train.py',
           os.path.join('config', config),
           f'--out_dir={current_model}']
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    lines = []
    for line in iter(process.stdout.readline, ''):
        if debug:
            print(line, end='', flush=True) 
        lines.append(line)
    
    process.wait()
    output = ''.join(lines)
    pattern = r'step (\d+): train loss ([\d.]+), val loss ([\d.]+)'
    match = re.search(pattern, output)
    if match:
        return {
            'steps': int(match.group(1)),
            'train_loss': float(match.group(2)),
            'val_loss': float(match.group(3))
        }
    else:
        return None, None, None



if __name__ == "__main__":
    kl_eval()
    ppl_eval()




