import sys, os, time
from dataclasses import asdict
import math
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from simpleGPT.Gpt2TrainConfig import Gpt2TrainConfig
from simpleGPT.simplegpt import GPT, GPTConfig
# from simpleGPT.SimpleDataLoader import SimpleDataloader
from simpleGPT.DistributedDataLoader import DistributedDataset, create_distributed_dataloader
import logging
import wandb
from tqdm import tqdm
from hellaswag import render_example, iterate_examples

assert os.getenv('BS') is not None


config = Gpt2TrainConfig(
    #Dataloader
    data_dir = "/workspaces/datasets/edu_fineweb10B",
    total_batch_size = 2048, # 2**19 # ~ 0.5M tokens
    bs = int(os.getenv('BS')),# 64 (A100 80Gb) 8 (RTX4090)
    
    # Model params
    block_size = 1024, #1024
    vocab_size = 50304, #50304
    n_layer = 12 ,#12
    n_head = 12, #12
    n_embd = 768, #768
    use_flash_attn = True, #True
    use_rope = False,
    
    # LR Scheduler params
    max_lr = 6e-4, #6e-4
    min_lr_ratio = 0.1, #0.1
    warmup_steps = 1, #GPT2:715 (100)
    max_steps = 10, #19_073
    val_every_n_steps = 2, #100
    val_n_steps = 2, #20
    
    # Optimizer
    wd = 0.1, #0.1
    
    # Other
    matmul_precision = 1, #1
    autocast_bf16 = False, #TRUE
    compile_model = False, #False
    use_grad_clip = True, #TRUE
    seed = 1337, #1337
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
    print(f"Using device: {device}")
device_type = "cuda" if device.startswith("cuda") else "cpu"

if master_process:
    print(f'### DDP enabled: {ddp}')
    print(f"### World size: {ddp_world_size}")
    print(f'### Init WandB run ...')
    wandb.init(
        project="SimpleGPT2_compare_impl",
        config = asdict(config)
    )
    print("### Configuration Parameters: %s", [f"{k}: {v} | " for k,v in asdict(config).items()])

### SEED
torch.manual_seed(config.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(config.seed)

### MATMUL PRECISION
matmul_prec_dict = {0:'medium', 1:'high', 2:'highest'}
torch.set_float32_matmul_precision(matmul_prec_dict[config.matmul_precision])

### LR COSINE SCHEDULER
min_lr = config.max_lr * config.min_lr_ratio
def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

### DATALOADER
if master_process: print("### Dataloaders...")
total_batch_size = config.total_batch_size
B,T = config.bs, config.block_size
assert total_batch_size % (B * T * ddp_world_size) == 0, total_batch_size / (B * T * ddp_world_size)
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    wandb.config['grad_accum_steps'] = grad_accum_steps
    wandb.config['ddp_world_size'] = ddp_world_size
    print(f"#### Total batch size: {total_batch_size} => gradient accumulation steps: {grad_accum_steps}")
train_loader = create_distributed_dataloader(config.data_dir, B=B, T=T, rank=ddp_rank, world_size=ddp_world_size, split='train')
val_loader = create_distributed_dataloader(config.data_dir, B=B, T=T, rank=ddp_rank, world_size=ddp_world_size, split='val')

### CREATE MODEL
if master_process: print("### Build Model...")
model = GPT(
    GPTConfig(
        block_size = config.block_size,
        vocab_size = config.vocab_size,
        n_layer = config.n_layer ,
        n_head = config.n_head,
        n_embd = config.n_embd,
        use_flash_attn = config.use_flash_attn,
        use_rope = config.use_rope
        )
    )
model.to(device)
if config.compile_model : model = torch.compile(model)
if ddp: model = DDP(module=model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model #Contains the model unwrapped 

### OPTIMIZER
if master_process: print("### Optimizer...")
optimizer = raw_model.configure_optimizers(weight_decay=config.wd , learning_rate=config.max_lr, device_type=device_type)

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

### TRAIN LOOP
if master_process: print("### Start trainning...")
dt_hist = []
train_iterator = iter(train_loader)
for step in range(config.max_steps):
    print("step",step)
    final_step = step == config.max_steps-1
    if master_process: t0 = time.time()
    
    ### VALIDATION
    if (step % config.val_every_n_steps == 0) or final_step:
        if master_process: print('### Validation')
        model.eval()
        val_iterator = iter(val_loader)
        with torch.no_grad():
            val_loss_accum = .0
            for i in range(min(config.val_n_steps, len(val_loader))):
                x, y = next(val_iterator)
                x, y = x.to(device), y.to(device)
                if config.autocast_bf16 :
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x, y)
                else:
                    logits, loss = model(x, y)
                loss = loss / config.val_n_steps
                val_loss_accum += loss.detach()
            
        if ddp: dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation loss: {val_loss_accum:04f}")
            wandb.log({"step": step, "val_loss": val_loss_accum})
    
    ### EVAL ON HELLASWAG
    if (not config.compile_model) and ((step % config.val_every_n_steps == 0) or final_step):
        if master_process: print('### Hellaswag evaluation')
        num_correct_norm = 0
        num_total = 0
        for i, example in tqdm(enumerate(iterate_examples("val")), total=10_042):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                if config.autocast_bf16 :
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(tokens)
                else:
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            wandb.log({'step':step, "HS_acc":acc_norm})

    ### TRAIN GRAD ACCUM LOOP
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = .0
    for accum_step in range(grad_accum_steps):
        x, y = next(train_iterator)
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (accum_step == grad_accum_steps - 1)
        if config.autocast_bf16 :
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        ## Scale the loss to compensate the gradient accumulation steps because the the loss is averaged
        # only across the bini-batch but not across the full accumulated batch
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    
    ### GRAD CLIP
    if config.use_grad_clip :
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    else:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        
    ### OPTIM STEP
    lr = get_lr(step, config.max_lr, min_lr, config.warmup_steps , config.max_steps)
    for param_group in optimizer.param_groups:
        param_group['lr']  = lr
    optimizer.step()
    
    ### LOG METRICS
    torch.cuda.synchronize()
    if ddp: dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    if master_process:
        t1 = time.time()
        dt = (t1-t0)
        dt_hist.append(dt)
        dt_hist = dt_hist[-20:]
        eta = (config.max_steps - step) * np.mean(dt_hist) / 60 / 60
        tokens_processed = B * T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        dtms = dt*1000
        print(f"Step {step}: loss: {loss_accum:.6f} | lr: {lr:4e} | norm: {norm:.4f} | dt: {dtms:.2f}ms | tok/s: {tokens_per_sec:.02f} | remain: {eta:.2f} h")
        wandb.log({
        "step": step,
        "train_loss": loss_accum,
        "lr": lr,
        "norm": norm,
        "dt_ms": dtms,
        "toks/s": tokens_per_sec
        })
        if master_process and (step % 5000 == 0) and step > 0:
            print("### Save checkpoint to WandB")
            model_artifact = wandb.Artifact(f'gpt-model-step-{step}', type='model')
            model_path = f'gpt-model-step-{step}.pth'
            torch.save(raw_model.state_dict(), model_path)
            model_artifact.add_file(model_path)
            wandb.log_artifact(model_artifact)

if master_process:
    print("### Save last checkpoint top WandB")
    model_artifact = wandb.Artifact(f'gpt-model-step-{step}', type='model')
    model_path = f'gpt-model-step-{step}.pth'
    torch.save(raw_model.state_dict(), model_path)
    model_artifact.add_file(model_path)
    wandb.log_artifact(model_artifact)
    wandb.finish()
    
if ddp: destroy_process_group()