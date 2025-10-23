import torch
import numpy as np
from model import GPT, GPTConfig
from tqdm import tqdm
import os

# from sample.py
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

# Load your model and data
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model = load_nanogpt_model('out-shakespeare-char-mps-8L-8H/ckpt.pt', device)

# Load data from val.bin
val_data = np.memmap('data/shakespeare_char/val.bin', dtype=np.uint16, mode='r')
encodings = torch.tensor(val_data, dtype=torch.long).unsqueeze(0)  # Add batch dimension

# Calculate perplexity using sliding window approach
max_length = model.config.block_size  # 256 for `config/train_shakespeare_char_mps.py`
stride = max_length // 2  # Half of block_size for good overlap
seq_len = encodings.size(1)  # Direct tensor access, no .input_ids

nll_sum = 0.0
n_tokens = 0
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings[:, begin_loc:end_loc].to(device)  # Direct tensor slicing
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        logits, loss = model(input_ids, target_ids)  # nanoGPT returns (logits, loss)
        neg_log_likelihood = loss

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
print(f"Perplexity: {ppl.item()}")