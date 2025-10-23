import subprocess
import sys
import time
import os
import argparse

def train_with_timeout(timeout_seconds, *args):
    cmd = ['python', 'train.py'] + list(args)
    start_time = time.time()
    process = subprocess.Popen(cmd)
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        print(f"\ntimed out after {timeout_seconds} s")
        process.terminate()
        process.wait()
    
    elapsed = time.time() - start_time
    print(f"total time: {elapsed:.1f} s")


def my_experiment_config(config_file, timeout_seconds=600):
    
    for layers, heads in [
        (2, 2), # 2 layers, 2 heads
        (2, 4),
        (4, 2),
        (4, 4),
        (4, 8),
        (8, 4),
        (8, 8)]:
        out_dir = f'out-shakespeare-char-mps-{layers}L-{heads}H'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args = [
            config_file,
            f'--n_layer={layers}',
            f'--n_head={heads}', 
            f'--out_dir={out_dir}'
        ]
        train_with_timeout(timeout_seconds, *args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/train_shakespeare_char_mps.py')
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    my_experiment_config(config_file=args.config, timeout_seconds=args.timeout)