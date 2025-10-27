import subprocess
import sys
import time
import os
import shutil
import argparse
import pickle
import json

debug = False

def finetune():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sourcemodel", required=False, type=str, default='out-shakespeare-char-mps-4L-2H',)
    parser.add_argument("--dataset", required=False, type=str, nargs='?', default=os.path.join('jane_austen_emma', 'emma_remap_vocab'))
    parser.add_argument("--finetunedmodel", required=False, type=str, nargs='?', default=None)
    parser.add_argument("--timeout","-t", required=False, type=int, default=1200)
    args = parser.parse_args()

    finetuned_model_dir = args.finetunedmodel
    if finetuned_model_dir is None:
        finetuned_model_dir = f"{args.sourcemodel}-finetuned"

    if os.path.exists(finetuned_model_dir):
        print("finetuned model dir already exists, continuing training")
    else:   
        os.makedirs(finetuned_model_dir)
        shutil.copy2(os.path.join(args.sourcemodel, 'ckpt.pt'),
                    os.path.join(finetuned_model_dir, 'ckpt.pt'))

    cmd = [
        sys.executable, 'train.py',
        'config/train_shakespeare_char_mps.py',
        f'--init_from=resume',
        f'--out_dir={finetuned_model_dir}', 
        f'--dataset={args.dataset}',
        f'--learning_rate=3e-4',
        f'--max_iters=3000',  # current model iterations + 1000
        f'--dropout=0.1'
    ]

    #. iter 3000: loss 0.8319, time 4301.80ms, mfu 0.16%
        
    start_time = time.time()
    process = subprocess.Popen(cmd)

    try:
        process.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait()
        
    elapsed = time.time() - start_time
    print(f"time: {elapsed:.1f}s")


def verify_embeddings(emma_ds = os.path.join('jane_austen_emma', 'emma_100pct')):

    with open(os.path.join('data', 'shakespeare_char', 'meta.pkl'), 'rb') as f:
        shakespeare_meta = pickle.load(f)
    
    with open(os.path.join('data', emma_ds, 'meta.pkl'), 'rb') as f:
        emma_meta = pickle.load(f)
    
    shakespeare_stoi = shakespeare_meta['stoi']
    emma_stoi = emma_meta['stoi']
    
    print(f"shakespeare vocab size: {len(shakespeare_stoi)}")
    print(f"emma vocab size: {len(emma_stoi)}")

    vocab_identical = shakespeare_stoi == emma_stoi
    if not vocab_identical:
        print("NOT IDENTICAL")
        for c in shakespeare_stoi:
            if c not in emma_stoi:
                print(f"missing in Emma: '{c}'")
        for c in emma_stoi:
            if c not in shakespeare_stoi:
                print(f"missing in Shakespeare: '{c}'")   
            # elif shakespeare_stoi[c] != emma_stoi[c]:
            #     print(f"different index for '{c}': Shakespeare={shakespeare_stoi[c]}, Emma={emma_stoi[c]}")
    else:
        print("IDENTICAL")

    return shakespeare_stoi, emma_stoi

def clean_emma_vocab(shakespeare_stoi, emma_stoi,
                     emma_old_input=os.path.join('jane_austen_emma', 'emma_100pct', 'emma100.txt'),
                     emma_new_ds=os.path.join('jane_austen_emma', 'emma_remap_vocab')):
    
    fn = os.path.basename(emma_old_input)

    shakespeare_missing = [c for c in emma_stoi if c not in shakespeare_stoi]
    replace_map = {c: '' for c in shakespeare_missing}
    replace_map.update({
        'é': 'e',
        'ê': 'e',
        'à': 'a',
        'ï': 'i',
        '‘': "'",
        '’': "'",
        '“': ''',
        '”': ''',
    })

    emma_missing = [c for c in shakespeare_stoi if c not in emma_stoi]

    with open(os.path.join('data', emma_old_input), 'r') as f:
        text = f.read()
    for c in replace_map:
        text = text.replace(c, replace_map[c])
    text = text + ' '.join(emma_missing)

    if not os.path.exists(os.path.join('data', emma_new_ds)):
        os.makedirs(os.path.join('data', emma_new_ds))

    with open(os.path.join('data', emma_new_ds, fn), 'w') as f:
        f.write(text)


    




if __name__ == "__main__":
    if debug:
        # shakespeare_stoi, emma_stoi = verify_embeddings(
        #     emma_ds=os.path.join('jane_austen_emma', 'emma_remap_vocab')
        # )
        shakespeare_stoi, emma_stoi = verify_embeddings()
        json.dump({'shakespeare_stoi': shakespeare_stoi, 'emma_stoi': emma_stoi}, open('vocab.json', 'w'), indent=4)
        clean_emma_vocab(shakespeare_stoi, emma_stoi)
    else:
        finetune()