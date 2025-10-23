import subprocess
import sys
import time
import os
import shutil

def finetune_model(
        source_model_dir,  # out-shakespeare-char-mps-8L-8H
        dataset,  # data/jane_austen_emma/train.bin
        finetuned_model_dir=None, # if not given will be {source_model_dir}-finetuned
        timeout_seconds=600):
    
    if finetuned_model_dir is None:
        finetuned_model_dir = source_model_dir.replace('shakespeare', 'finetuned')
    if os.path.exists(finetuned_model_dir):
        print(f"dir ({finetuned_model_dir}) exists, deleting")
        shutil.rmtree(finetuned_model_dir)
    
    os.makedirs(finetuned_model_dir)
    shutil.copy2(os.path.join(source_model_dir, 'ckpt.pt'),
                 os.path.join(finetuned_model_dir, 'ckpt.pt'))

    # Now resume training from the copied checkpoint
    cmd = [
        sys.executable, 'train.py',
        'config/train_shakespeare_char_mps.py',
        f'--init_from=resume',
        f'--out_dir={finetuned_model_dir}',  # Load from and save to the new directory
        f'--dataset={dataset}',
        f'--learning_rate=3e-4',
        f'--max_iters=2000',
        f'--dropout=0.1'
    ]
    
    start_time = time.time()
    process = subprocess.Popen(cmd)
    
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait()
    
    elapsed = time.time() - start_time
    print(f"Time: {elapsed:.1f}s")

if __name__ == "__main__":
    source_dir = sys.argv[1]
    target_dataset = sys.argv[2] if len(sys.argv) > 2 else 'jane_austen_emma'
    output_dir = sys.argv[3] if len(sys.argv) > 3 else source_dir.replace('shakespeare', 'finetuned')
    
    finetune_model(source_dir, target_dataset, output_dir)
