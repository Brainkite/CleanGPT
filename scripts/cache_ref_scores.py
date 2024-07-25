import sys, os, time
from dataclasses import asdict
import math
import torch
import numpy as np
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group, barrier
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from simpleGPT.Gpt2TrainConfig import Gpt2TrainConfig
from simpleGPT.simplegpt import GPT, GPTConfig
from simpleGPT.JestDistributedDataLoader import JestDistributedDataloader
import logging
import wandb
from tqdm import tqdm
from hellaswag import render_example, iterate_examples

assert os.getenv('BS') is not None

config = Gpt2TrainConfig(
    #Dataloader
    data_dir = "/workspace/datasets/edu_fineweb10B",
    total_batch_size = 2**19, # 2**19 # ~ 0.5M tokens
    bs = int(os.getenv('BS')),# 64 (A100 80Gb) 8 (RTX4090)
    shuffle_seq= False,
    
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
    warmup_steps = 100, #GPT2:715 (100)
    max_steps = 19_073, #19_073
    val_every_n_steps = 250, #100
    val_n_steps = 20, #20
    
    # Optimizer
    wd = 0.1, #0.1
    
    # Other
    matmul_precision = 1, #1
    autocast_bf16 = True, #TRUE
    compile_model = True, #False
    use_grad_clip = True, #TRUE
    seed = 1337, #1337
    
    #JEST
    ref_model_name = 'openai-community/gpt2-medium',
    online_jest = False,
    filtering_ratio = 0.8,
    n_chunks = 16,
    ref_scores_fp='/workspace/datasets/ref_scores/edu_fineweb10B_ref_scores_gpt2-medium_T1024.npy'
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
    config.ref_scores_fp = config.ref_scores_fp[:-4] + f'_rank{ddp_rank}.npy'
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
    print("### Configuration Parameters: %s", [f"{k}: {v} | " for k,v in asdict(config).items()])

### MATMUL PRECISION
matmul_prec_dict = {0:'medium', 1:'high', 2:'highest'}
torch.set_float32_matmul_precision(matmul_prec_dict[config.matmul_precision])

train_loader = JestDistributedDataloader(
    config.data_dir, 
    config.bs, 
    config.block_size, 
    process_rank=ddp_rank, 
    num_processes=ddp_world_size, 
    split='train', 
    shuffle=False,
    jest=True,
    ref_scores_fp=config.ref_scores_fp
)

if master_process:
    print("Computing and caching reference scores...")

train_loader.compute_and_cache_ref_scores(config.ref_model_name, device, device_type, config.autocast_bf16, config.compile_model)

if master_process:
    print("Finished computing and caching reference scores.")

if ddp:
    barrier()
    destroy_process_group()

if master_process and ddp:
    print("Merging reference scores from all ranks...")
    
    # Get the base filename without rank information
    base_filename = config.ref_scores_fp[:-10] + '.npy'  # Remove '_rankX.npy'
    
    all_scores = []
    for rank in range(ddp_world_size):
        rank_filename = base_filename[:-4] + f'_rank{rank}.npy'
        if os.path.exists(rank_filename):
            rank_scores = np.load(rank_filename)
            all_scores.append(rank_scores)
            os.remove(rank_filename)  # Remove the rank-specific file
        else:
            print(f"Warning: File for rank {rank} not found: {rank_filename}")
    
    if all_scores:
        combined_scores = np.concatenate(all_scores)
        
        # Sort the combined scores
        sort_idxs = np.lexsort((combined_scores['sample_idx'], combined_scores['shard_idx']))
        sorted_scores = combined_scores[sort_idxs]
        
        # Save the sorted, combined scores
        np.save(base_filename, sorted_scores)
        print(f"Combined reference scores saved to: {base_filename}")
    else:
        print("No scores were found to merge.")

print("Script completed.")
    
    
    
    