import sys, os, time
import math
import torch
import torch.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from simpleGPT.simplegpt import GPT, GPTConfig
from simpleGPT.DistributedDataLoader import DistributedDataloader
import logging
import wandb
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = dict(
    #Dataloader
    data_dir = "/workspace/datasets/edu_fineweb10B",
    total_batch_size = 2**19, # 2**19 # ~ 0.5M tokens
    bs = os.getenv('BS'), # 64 (A100 80Gb)
    
    # Model params
    block_size = 1024, #1024
    vocab_size = 50304, #50304
    n_layer = 12, #12
    n_head = 12, #12
    n_embd = 768, #768
    use_flash_attn = True, #True
    
    # LR Scheduler params
    max_lr = 6e-4, #6e-4
    min_lr_ratio = 0.1, #0.1
    warmup_steps = 100, #GPT2:715 (200)
    max_steps = 19_073, #19_073
    
    # Optimizer
    wd = 0.1, #0.1
    
    # Other
    matmul_precision = 1, #1
    autocast_bf16 = True, #TRUE
    compile_model = True, #TRUE
    use_grad_clip = True, #TRUE
    seed = 1337 #1337
)

# SETUP DDP
ddp = int(os.environ.get('RANK', -1)) != -1 # Check if current process is in ddp run
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    logger.info(f"Using device: {device}")
device_type = "cuda" if device.startswith("cuda") else "cpu"

if master_process:
    logger.info(f'### DDP enabled: {ddp}')
    logger.info(f"### World size: {ddp_world_size}")
    wandb.init(
        project="SimpleGPT2_FWedu10B_train",
        config = config
    )
    logger.info("### Configuration Parameters: %s", [f"{k}: {v} | " for k,v in config.items()])

### SEED
torch.manual_seed(int(config['seed']))
if torch.cuda.is_available(): torch.cuda.manual_seed(int(config['seed']))

### MATMUL PRECISION
matmul_prec_dict = {0:'medium', 1:'high', 2:'highest'}
torch.set_float32_matmul_precision(matmul_prec_dict[int(config['matmul_precision'])])

### LR COSINE SCHEDULER
min_lr = float(config['max_lr']) * float(config['min_lr_ratio'])
def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):
    if it < warmup_steps:
        return (it+1)/warmup_steps * max_lr
    
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

### DATALOADER
total_batch_size = config['total_batch_size']
B,T = config['bs'], config['block_size']
assert total_batch_size % (B * T * ddp_world_size) == 0, total_batch_size / (B * T * ddp_world_size)
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    wandb.config['grad_accum_steps'] = grad_accum_steps
    wandb.config['ddp_world_size'] = ddp_world_size
    logger.info(f"### Total batch size: {total_batch_size} => gradient accumulation steps: {grad_accum_steps}")
train_loader = DistributedDataloader(config["data_dir"], B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')

### CREATE MODEL
if master_process: logger.info("### Build Model...")
model = GPT(
    GPTConfig(
        block_size = int(T),
        vocab_size = int(config['vocab_size']),
        n_layer = int(config['n_layer']),
        n_head = int(config['n_head']),
        n_embd = int(config['n_embd']),
        use_flash_attn = bool(config['use_flash_attn']), 
        )
    )
model.to(device)
if config['compile_model']: model = torch.compile(model)
if ddp: model = DDP(module=model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model #Contains the model unwrapped 

### OPTIMIZER
optimizer = raw_model.configure_optimizers(weight_decay=config['wd'], learning_rate=float(config['max_lr']), device=device)

### TRAIN LOOP
if master_process: logger.info("### Start trainning...")
for step in range(int(config['max_steps'])):
    if master_process: t0 = time.time()
    optimizer.zero_grad()
    
    ### GRAD ACCUM LOOP
    loss_accum = .0
    for accum_step in range(int(grad_accum_steps)):
        x, y = train_loader.new_batch()
        x, y = x.to(device), y.to(device)
        if config['autocast_bf16']:
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        ## Scale the loss to compensate the gradient accumulation steps because the the loss is averaged
        # only across the bini-batch but not across the full accumulated batch
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            # loss.backward will deposit grads AND sync across GPUs
            # so we disable grad sync untill last accum_step
            model.require_backward_grad_sync = (accum_step == grad_accum_steps-1)
        loss.backward()
    
    ### GRAD CLIP
    if config['use_grad_clip']:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    else:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        
    ### OPTIM STEP
    lr = get_lr(step, float(config['max_lr']), min_lr, config['warmup_steps'], config['max_steps'])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    
    ### LOG METRICS
    torch.cuda.synchronize()
    if ddp: dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
        t1 = time.time()
        dt = (t1-t0)
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        dtms = dt*1000
        logger.info(f"Step {step}: loss: {loss_accum:.6f} | lr: {lr:4e} | norm: {norm:.4f} | dt: {dtms:.2f}ms | tok/s: {tokens_per_sec:.02f} | toks: {tokens_processed}")
        wandb.log({
        "step": step,
        "loss": loss_accum,
        "lr": lr,
        "norm": norm,
        "dt_ms": dtms,
        "toks/s": tokens_per_sec
        })
        
if ddp: destroy_process_group()