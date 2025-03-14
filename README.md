# SimpleGPT

A clean PyTorch implementation of GPT-2 for pre-training on the edu_fineweb10B dataset. This repository provides a straightforward and educational reproduction of the GPT-2 architecture with modern PyTorch features.

## Features

- 🚀 Clean, modular implementation of GPT-2 architecture
- ⚡ Efficient training with Flash Attention support
- 🔄 Distributed training support with DDP (DistributedDataParallel)
- 📊 Integration with Weights & Biases for experiment tracking
- 🎯 Support for loading pre-trained GPT-2 weights
- 🛠️ Configurable model sizes matching OpenAI's GPT-2 variants

## Installation

```bash
git clone https://github.com/yourusername/simpleGPT.git
cd simpleGPT
pip install -e .
```

## Requirements

```
transformers[torch]
pytest
tiktoken
wandb
```

## Training

The training script (`scripts/train.py`) supports:
- Gradient accumulation for large batch sizes
- Learning rate scheduling with warmup
- Distributed training across multiple GPUs
- Weight decay with AdamW optimizer
- Mixed precision training with bfloat16

Example training command:
```bash
python scripts/train.py
```

For distributed training:
```bash
torchrun --nproc_per_node=N scripts/train.py
```

## Configuration

The model and training parameters can be configured through the config dictionary in `scripts/train.py`. Key parameters include:

```python
config = dict(
    # Model architecture
    block_size = 1024,
    vocab_size = 50304,
    n_layer = 12,
    n_head = 12,
    n_embd = 768,
    
    # Training
    total_batch_size = 2**19,
    max_lr = 6e-4,
    min_lr_ratio = 0.1,
    warmup_steps = 10,
    max_steps = 100,
)
```