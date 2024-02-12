import time

out_dir = 'out-hi-100k'
eval_interval = 100 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often
wandb_log = False # feel free to turn on
wandb_project = 'shakespeare'
wandb_run_name = 'ft-' + str(time.time())

dataset = 'madhushala'
init_from = 'resume' 

# only save checkpoints if the validation loss improves
always_save_checkpoint = True

batch_size = 128
block_size = 256 # context of up to 256 previous characte
gradient_accumulation_steps = 1
max_iters = 20

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# finetune at constant LR
learning_rate = 3e-6
max_iters = 10_000
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially
decay_lr = False
