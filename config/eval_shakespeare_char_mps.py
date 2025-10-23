batch_size = 8
eval_iters = 500
eval_only = True
wandb_log = False
init_from = 'resume'

always_save_checkpoint = False
block_size = 256
dataset = 'shakespeare_char'

device = 'mps'  # run on cpu only
compile = False # do not torch compile the model


# Usage
# 
