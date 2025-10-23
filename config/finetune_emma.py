# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'placeholder'
eval_interval = 50 # keep frequent because we'll overfit
eval_iters = 50
log_interval = 5 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'project'
wandb_run_name = 'run'

init_from = 'resume'
dataset = 'jane_austen_emma'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1

# learning_rate = 1e-3 # with baby networks can afford to go a bit higher
# max_iters = 5000
# lr_decay_iters = 5000 # make equal to max_iters usually
learningg_rate = 3e-4  # Lower learning rate for finetuning
max_iters = 2000      # Fewer iterations for finetuning
lr_decay_iters = 2000

min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'mps'  # run on cpu only
compile = False # do not torch compile the model
