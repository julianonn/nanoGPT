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

def kl_eval(input_data, 
            model_directories, 
            ngram=3, 
            device='mps', 
            output_json='kl_eval.json'):
    
    print("----- KL DIVERGENCE -----")
    # read input file
    with open(input_data, 'r') as f:
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


def get_losses(current_model, timeout_seconds=1800, debug=True):
    cmd = ['python', 'train.py',
           'config/eval_shakespeare_char_mps.py',
           f'--out_dir={current_model}']
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    lines = []
    for line in iter(process.stdout.readline, ''):
        if debug:
            print(line, end='', flush=True) 
        lines.append(line)
    
    process.wait()
    output = ''.join(lines)
    if debug:
        with open('losses.txt', 'w') as f:
            f.write(output)
    return output
    

def ppl_eval(test_data, 
            model_directories, 
            device='mps', 
            output_json='ppl_eval.json'):
    
    print("----- PERPLEXITY -----")



    # Load data from val.bin
    val_data = np.memmap(test_data, dtype=np.uint16, mode='r')
    encodings = torch.tensor(val_data, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    
    res = []
    for d in model_directories:
        # try:
            print(f'evaluating model in {d}')
            model = load_model(out_dir=d, device=device)
            ppl_scalar = hf_perplexity(model, encodings, device=device)
            res.append({
                'model_dir': d,
                'perplexity': ppl_scalar,
            })
        # except Exception as e:
        #     print(f"error, skipping model")

    print('writing results to', output_json)
    with open(output_json, 'w') as f:
        tmp = {'time': time.time(), 'results': res}
        json.dump(tmp, f, indent=4)
    return res
    
    
def load_model(out_dir = 'out-shakespeare-char-mps', device='mps'):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model

ce_lose
    

def hf_perplexity(model, encodings, device='mps'):
    # based on https://huggingface.co/docs/transformers/perplexity
    # slightly modified

    # Calculate perplexity using sliding window approach
    max_length = model.config.block_size  # 256 for `config/train_shakespeare_char_mps.py`
    stride = max_length // 2  # Half of block_size for good overlap
    seq_len = encodings.size(1)  # Direct tensor access, no .input_ids

    print(f"max_length: {max_length}  seq_len: {seq_len}   stride: {stride}")

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    # iter = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        # iter += 1
        # print(f"iter {iter}/{seq_len // stride}", flush=True)
        
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        # input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        input_ids = encodings[:, begin_loc:end_loc].to(device)  # Direct tensor slicing
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            # outputs = model(input_ids, labels=target_ids)
            logits, neg_log_likelihood = model(input_ids, target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            # neg_log_likelihood = outputs.loss

        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)
    return ppl.item()  # tensor --> scalar



def get_losses(model_directories, device='mps', output_json='train_val_losses.json'):
    res = []
    for d in model_directories:
        ckpt_path = os.path.join(d, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        config = checkpoint.get('config', {}) 
        # print(checkpoint)
        res.append({
            'model_dir': d,
            # 'train_loss': config.get('train_loss').item(),
            'val_loss': checkpoint.get('best_val_loss').item(),
            'iter_num': checkpoint.get('iter_num', None)
        })
    
    with open(output_json, 'w') as f:
        tmp = {'time': time.time(), 'results': res}
        json.dump(tmp, f, indent=4)

if __name__ == "__main__":
    get_losses(model_directories=SHAKESPEARE_CHAR_DIRS)
    # kl_eval(input_data='data/shakespeare_char/input.txt',
    #        model_directories=SHAKESPEARE_CHAR_DIRS)
   
    # ppl_eval(test_data='data/shakespeare_char/val.bin', 
    #         model_directories=SHAKESPEARE_CHAR_DIRS)




