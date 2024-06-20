import sys, os, time
from dataclasses import dataclass, asdict
import math
import torch
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from simpleGPT.simplegpt import GPT, GPTConfig
from simpleGPT.DistributedDataLoader import DistributedDataloader
import logging
import wandb
from hellaswag import render_example, iterate_examples
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

assert os.getenv('BS') is not None

@dataclass
class Gpt2TrainConfig:
    #Dataloader
    data_dir = "/workspace/datasets/edu_fineweb10B"
    total_batch_size = 2**19 # 2**19 # ~ 0.5M tokens
    bs = int(os.getenv('BS')) # 64 (A100 80Gb) 8 (RTX4090)
    
    # Model params
    block_size = 1024 #1024
    vocab_size = 50304 #50304
    n_layer = 12 #12
    n_head = 12 #12
    n_embd = 768 #768
    use_flash_attn = True #True
    
    # LR Scheduler params
    max_lr = 6e-4 * 3 #6e-4
    min_lr_ratio = 0.1 #0.1
    warmup_steps = 100 #GPT2:715 (100)
    max_steps = 19_073 * 2 #19_073
    val_every_n_steps = 100 #100
    val_n_steps = 20 #20
    
    # Optimizer
    wd = 0.1 #0.1
    
    # Other
    matmul_precision = 1 #1
    autocast_bf16 = True #TRUE
    compile_model = False #False
    use_grad_clip = True #TRUE
    seed = 1337 #1337

config = Gpt2TrainConfig()

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
        config = asdict(config)
    )
    logger.info("### Configuration Parameters: %s", [f"{k}: {v} | " for k,v in asdict(config).items()])

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
        return (it+1)/warmup_steps * max_lr
    
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

### DATALOADER
total_batch_size = config.total_batch_size
B,T = config.bs, config.block_size
assert total_batch_size % (B * T * ddp_world_size) == 0, total_batch_size / (B * T * ddp_world_size)
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    wandb.config['grad_accum_steps'] = grad_accum_steps
    wandb.config['ddp_world_size'] = ddp_world_size
    logger.info(f"### Total batch size: {total_batch_size} => gradient accumulation steps: {grad_accum_steps}")
train_loader = DistributedDataloader(config.data_dir, B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = DistributedDataloader(config.data_dir, B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

### CREATE MODEL
if master_process: logger.info("### Build Model...")
model = GPT(
    GPTConfig(
        block_size = config.block_size,
        vocab_size = config.vocab_size,
        n_layer = config.n_layer ,
        n_head = config.n_head,
        n_embd = config.n_embd,
        use_flash_attn = config.use_flash_attn, 
        )
    )
model.to(device)
if config.compile_model : model = torch.compile(model)
if ddp: model = DDP(module=model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model #Contains the model unwrapped 

### OPTIMIZER
optimizer = raw_model.configure_optimizers(weight_decay=config.wd , learning_rate=config.max_lr, device=device)

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
if master_process: logger.info("### Start trainning...")
for step in range(config.max_steps):
    if master_process: t0 = time.time()
    
    ### VALIDATION
    if (step+1 % config.val_every_n_steps == 0) or (step==config.max_steps-1):
        if master_process: logger.info('### Validation')
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = .0
            for _ in range(config.val_n_steps):
                x, y = val_loader.new_batch()
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
            logger.info(f"Validation loss: {val_loss_accum:04f}")
            wandb.log({"step": step, "loss": val_loss_accum})
    
    ### EVAL ON HELLASWAG
    if (step+1 % config.val_every_n_steps == 0 or (step==config.max_steps-1)) and (not config.compile_model):
        if master_process: logger.info('### Hellaswag evaluation')
        num_correct_norm = 0
        num_total = 0
        for i, example in zip(range(5000), iterate_examples("val")):
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
    optimizer.zero_grad()
    loss_accum = .0
    for accum_step in range(int(grad_accum_steps)):
        x, y = train_loader.new_batch()
        x, y = x.to(device), y.to(device)
        if config.autocast_bf16 :
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
    if config.use_grad_clip :
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    else:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        
    ### OPTIM STEP
    lr = get_lr(step, config.max_lr, min_lr, config.warmup_steps , config.max_steps )
    for param_group in optimizer.param_groups:
        param_group['lr']  = lr
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
        if master_process and (step % 5000 == 0):
            logger.info("### Save checkpoint to WandB")
            model_artifact = wandb.Artifact(f'gpt-model-step-{step}', type='model')
            model_path = f'gpt-model-step-{step}.pth'
            torch.save(raw_model.state_dict(), model_path)
            model_artifact.add_file(model_path)
            wandb.log_artifact(model_artifact)

if master_process:
    logger.info("### Save last checkpoint top WandB")
    model_artifact = wandb.Artifact(f'gpt-model-step-{step}', type='model')
    model_path = f'gpt-model-step-{step}.pth'
    torch.save(raw_model.state_dict(), model_path)
    model_artifact.add_file(model_path)
    wandb.log_artifact(model_artifact)
    
if ddp: destroy_process_group()