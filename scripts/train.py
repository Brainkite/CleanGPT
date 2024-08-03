import os, time
from dataclasses import asdict
import torch
import numpy as np
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.autograd.profiler as profiler
import torch.distributed as dist
from simpleGPT.Gpt2TrainConfig import Gpt2TrainConfig
from simpleGPT.simplegpt import GPT, GPTConfig
from simpleGPT.JestDistributedDataLoader import JestDistributedDataloader
import wandb
from simpleGPT.trainer import hellaswag_eval_step, jest_train_step, validation_step

# assert os.getenv('BS') is not None

config = Gpt2TrainConfig(
    #Dataloader
    data_dir = "/workspace/datasets/edu_fineweb10B",
    total_batch_size = 16 * 1024, # 2**19 # ~ 0.5M tokens
    # bs = int(os.getenv('BS')),# 64 (A100 80Gb) 8 (RTX4090)
    bs = 2,
    shuffle_seq= True,
    
    # Model params
    block_size = 1024, #1024
    vocab_size = 50304, #50304
    n_layer = 12 ,#12
    n_head = 12, #12
    n_embd = 768, #768
    use_flash_attn = True, #True
    use_rope = False,
    
    # LR Scheduler params
    max_lr = 6e-4 * 3, #6e-4
    min_lr_ratio = 0.1, #0.1
    warmup_steps = 2, #GPT2:715 (100)
    max_steps = 1, #19_073
    val_every_n_steps = 2, #100
    val_n_steps = 2, #20
    
    # Optimizer
    wd = 0.1, #0.1
    
    # Other
    matmul_precision = 1, #1
    autocast_bf16 = True, #TRUE
    compile_model = False, #False
    use_grad_clip = True, #TRUE
    seed = 1337, #1337
    
    #JEST
    ref_model_name = 'openai-community/gpt2',
    online_jest = False,
    filtering_ratio = 0.5,
    n_chunks = 4,
    ref_scores_fp='/workspace/datasets/ref_scores/edu_fineweb10B_ref_scores_gpt2_T512.npyyyy'
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
    wandb.init(
        project="SimpleGPT2_compare_impl",
        config = asdict(config)
    )
    print("### Configuration Parameters: %s", [f"\n{k}: {v}" for k,v in asdict(config).items()])

### SEED
torch.manual_seed(config.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(config.seed)

### MATMUL PRECISION
matmul_prec_dict = {0:'medium', 1:'high', 2:'highest'}
torch.set_float32_matmul_precision(matmul_prec_dict[config.matmul_precision])

### DATALOADER
total_batch_size = config.total_batch_size
B,T = config.bs, config.block_size

super_batch_size = int(config.total_batch_size / (1 - config.filtering_ratio))
super_batch_size = (super_batch_size // (B * T * ddp_world_size)) * (B * T * ddp_world_size)
actual_filtering_ratio = 1 - (config.total_batch_size / super_batch_size)
jest_steps = super_batch_size // (B * T * ddp_world_size)

assert total_batch_size % (B * T * ddp_world_size) == 0, total_batch_size / (B * T * ddp_world_size)
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    wandb.config['grad_accum_steps'] = grad_accum_steps
    wandb.config['ddp_world_size'] = ddp_world_size
    wandb.config['super_batch_size'] = super_batch_size
    wandb.config['actual_filtering_ratio'] = actual_filtering_ratio
    print(f"\n### Total batch size: {total_batch_size//T} samples => gradient accumulation steps: {grad_accum_steps}")
    print(f"### Super batch size: {super_batch_size//T} samples => jest steps: {jest_steps}")
    print(f"### Actual filtering ratio: {actual_filtering_ratio:.4f}")

train_loader = JestDistributedDataloader(
    config.data_dir, 
    B, T, 
    jest=True, 
    process_rank=ddp_rank, 
    num_processes=ddp_world_size, 
    split='train', 
    shuffle=config.shuffle_seq,
    ref_scores_fp=config.ref_scores_fp
    )

val_loader = JestDistributedDataloader(
    config.data_dir, 
    B, T, 
    jest=False, 
    process_rank=ddp_rank, 
    num_processes=ddp_world_size, 
    split='val', 
    shuffle=False
    )

def _jest_forward(x,y,model):
    logits, _ = model(x, y)
    # Compute per-sample loss
    probs = logits.view(-1, config.vocab_size)
    labs = y.flatten()
    samples_loss = torch.nn.functional.cross_entropy(
        probs,
        labs,
        reduction='none'
    )
    samples_loss = samples_loss.view(y.size()).mean(dim=1)
    return samples_loss

def joint_examples_selection(learn_scores, n_examples, n_chunks):
    assert n_examples % n_chunks == 0
    chunk_size = n_examples // n_chunks
    selected_indices = []
    available_indices = np.arange(len(learn_scores))
    
    for _ in range(n_chunks):
        available_scores = learn_scores[available_indices]
        chunk_indices = np.argpartition(available_scores, -chunk_size)[-chunk_size:]
        chunk_indices = chunk_indices[np.argsort(available_scores[chunk_indices])[::-1]]
        selected_indices.extend(available_indices[chunk_indices])
        available_indices = np.delete(available_indices, chunk_indices)
        
    assert len(selected_indices) == n_examples
    return selected_indices

### CREATE MODEL
if master_process: print("\n### Build Model...")
model = GPT(
    GPTConfig(
        block_size = config.block_size,
        vocab_size = config.vocab_size,
        n_layer = config.n_layer ,
        n_head = config.n_head,
        n_embd = config.n_embd,
        use_flash_attn = config.use_flash_attn,
        use_rope= config.use_rope
        )
    )
model.to(device)
# TODO: add suupot to keep compiled and uncompiled model in memory for faster train and validation taime but stable hellaswag eval
if config.compile_model : model = torch.compile(model)
if ddp: model = DDP(module=model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model #Contains the model unwrapped 

### OPTIMIZER
optimizer = raw_model.configure_optimizers(weight_decay=config.wd , learning_rate=config.max_lr, device_type=device_type)

### TRAIN LOOP
if master_process: print("\n### Start trainning...")
dt_hist = []
for step in range(config.max_steps):
    final_step = step == config.max_steps-1
    if master_process: t0 = time.time()
    
    ### VALIDATION
    if step>0 and (step % config.val_every_n_steps == 0) or final_step:
        if master_process: print('\n### Validation')
        val_loss_accum = validation_step(config, device, device_type, val_loader, model)
        if ddp: dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation loss: {val_loss_accum:04f}")
            wandb.log({"step": step, "val_loss": val_loss_accum})
    
    ### EVAL ON HELLASWAG
    if step>0 and (not config.compile_model) and ((step % config.val_every_n_steps == 0) or final_step):
        if master_process: print('\n### Hellaswag evaluation')
        num_correct_norm, num_total, acc_norm = hellaswag_eval_step(config, ddp, ddp_rank, ddp_world_size, device, device_type, model)
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            wandb.log({'step':step, "HS_acc":acc_norm})

    ## JEST Example Selection
    if master_process: print('\n### JEST Example Selection')
    model.eval()
    dtype = np.dtype([('shard_idx', np.uint16), ('sample_idx', np.uint32), ('learn_score', np.float32)])
    scores = np.zeros(super_batch_size // (T * ddp_world_size), dtype=dtype)
    for i in range(jest_steps):
        x, y, ref_scores = train_loader.new_batch()
        x, y = x.to(device), y.to(device)
        
        if config.autocast_bf16 :
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                samples_losses = _jest_forward(x,y,model)
        else:
            samples_losses = _jest_forward(x,y,model)
        
        learn_score = samples_losses.detach().cpu().numpy() - ref_scores['score']
        
        start = i * B
        end = start + B
        scores[['shard_idx', 'sample_idx']][start: end] = ref_scores[['shard_idx', 'sample_idx']]
        scores['learn_score'][start: end] = learn_score
    
    torch.cuda.synchronize()
    if ddp:
        gathered_scores = [np.empty(0) for _ in range(ddp_world_size)]
        dist.all_gather_object(gathered_scores, scores)
        scores = np.concatenate(gathered_scores)
    
    if master_process: print(f'### {len(scores)} scores gathered')
    
    selected_idxs = joint_examples_selection(scores['learn_score'], total_batch_size//T, config.n_chunks)
    selected_idxs = list(np.array_split(selected_idxs , ddp_world_size)[ddp_rank])
    examples_idxs = scores[['shard_idx','sample_idx']][selected_idxs]
    if master_process: print(f'### Slected {len(examples_idxs)} examples in {config.n_chunks} chunks for each rank')
    
    subset_dataloader = train_loader.new_dataloader_from_idxs(examples_idxs)

    ### TRAIN GRAD ACCUM LOOP
    grad_accum_steps = len(subset_dataloader)
    if master_process: print(f'\n### Train step with {grad_accum_steps} grad accum steps')
    loss_accum, norm, lr = jest_train_step(config, ddp, device, device_type, model, optimizer, step, subset_dataloader, grad_accum_steps)
    
    ## LOG STATS
    if master_process:
        t1 = time.time()
        dt = (t1-t0)
        dt_hist.append(dt)
        dt_hist = dt_hist[-20:]
        eta = (config.max_steps - step) * np.mean(dt_hist) / 60 / 60
        tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
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
        if master_process and (step % 5000 == 0) and (step > config.max_steps//2):
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