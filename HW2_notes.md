# Code corresponding to HW2


**Hardware:** MacBook Air 2022 (M2 Chip, 16GB RAM, 256GB SSD)

## Question 2

### 2.3  Hyperparameter Experimentation

By default, the CPU training script uses `n_layer=4` and `n_head=4` and maxes out at 2000 iterations. Results:
- Loss: 1.6958
- Time: 308.70ms
- MFU: 0.05%

I wrote a wrapper script, `q2_train_timeout.py` that times out after 10 minutes instead. Usage (for experiment):

```
python3 q2_train_notimeout.py --timeout 600 --config config/train_shakespeare_char_mps.py
```

### 2.4  Evaluation metrics

See `q2_evaluation.py`. Usage: `python3 q2_evaluation.py`

This script dumps KL divergence and Perplexity outputs to  `kl_eval.json` and `ppl_eval.json` respectively.


### Dataset
Jane Austen's Emma from Project Gutenberg ([https://www.gutenberg.org/ebooks/158]). See `data/jane_austen_emma/emma.txt` for cleaned dataset. Run `python data/jane_austen_emma/prepare.py` to generate train and test sets, and `q2_finetune.py` for finetuning scripts.


