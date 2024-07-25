from dataclasses import dataclass

@dataclass
class Gpt2TrainConfig:
    #Dataloader
    data_dir: str
    total_batch_size: int
    bs: int
    shuffle_seq: bool
    
    # Model params
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    use_flash_attn: bool
    use_rope: bool
    
    # LR Scheduler params
    max_lr: float
    min_lr_ratio: float
    warmup_steps: int
    max_steps: int
    val_every_n_steps: int
    val_n_steps: int
    
    # Optimizer
    wd: float
    
    # Other
    matmul_precision: int
    autocast_bf16: bool
    compile_model: bool
    use_grad_clip: bool
    seed: int
    
    #JEST
    ref_model_name: str
    online_jest: bool
    filtering_ratio: float
    n_chunks: int
    ref_scores_fp: str
    