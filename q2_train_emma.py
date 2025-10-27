import subprocess
import sys
import time
import os
import json

from q2_train_timeout import train_with_timeout
from q2_evaluation import ppl_eval, kl_eval


EMMA_DIRS = [
    'emma_25pct',
    'emma_50pct',
    'emma_75pct',
    'emma_100pct'
]

TIMEOUT = 1200

EMMA_OUTPUT_FILE = 'eval_emma.json'
results = { d: {} for d in EMMA_DIRS }

for dir in EMMA_DIRS:
    print(f"\n------  training {dir}  ------")
    dataset = os.path.join('jane_austen_emma', f'{dir}')
    out_dir = f'out-emma-char-mps-{dir}'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
        cmd = ['python', 
            'train.py', 
            'config/train_emma_char_mps.py',
            f'--dataset={dataset}',
            f'--out_dir={out_dir}']
        
        start_time = time.time()
        process = subprocess.Popen(cmd)
        try:
            process.wait(timeout=TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"\ntimed out after {TIMEOUT} s")
            process.terminate()
            process.wait()
        process.wait()
        elapsed = time.time() - start_time
        print(f"total time: {elapsed:.1f} s")

        results[dir]['training_time'] = elapsed

with open(EMMA_OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=4)


# calculate KL divergence 
for pct in [25, 50, 75, 100]:
# for pct in [100]:
    dir = f'emma_{pct}pct'
    kl_res = kl_eval(
        input_data = os.path.join('jane_austen_emma', dir, f'emma{pct}.txt'),
        model_directories = [f'out-emma-char-mps-{dir}'],
        write=False)
    kl = kl_res[0].get('kl_div_loss')

    ppl_res = ppl_eval(
        model_directories = [f'out-emma-char-mps-{dir}'],
        config = 'eval_emma_char_mps.py',
        write=False)

    results[dir].update({
        'kl_divergence': kl,
        'steps': ppl_res[0].get('steps'),
        'train_loss': ppl_res[0].get('train_loss'),
        'val_loss': ppl_res[0].get('val_loss'),
        'train_ppl': ppl_res[0].get('train_ppl'),
        'val_ppl': ppl_res[0].get('val_ppl')
    })

with open(EMMA_OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=4)


