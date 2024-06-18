import torch
import torch.functional as F
from simpleGPT.simplegpt import GPT, GPTConfig
from simpleGPT.simpleDataLoader import SimpleDataloader
import time
import math

#Inference params
num_return_seq = 5
max_length = 30

# Model params
B,T = 2, 512
device = 'cuda'
torch.set_float32_matmul_precision('high')
autocast_bf16=False

# LR Scheduler parames
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)
    
def get_lr(it):
    if it < warmup_steps:
        return (it+1)/warmup_steps * max_lr
    
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

train_loader = SimpleDataloader("dataset/tiny_shakesprear.txt", B, T)

model = GPT(GPTConfig())
model.to(device)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)
model = torch.compile(model)

for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.new_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    if autocast_bf16:
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
    else:
        logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)*1000
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1-t0)
    print(f"Step {step}: loss: {loss.item():.6f}; lr: {lr:4e}; norm: {norm:.4f}; dt: {dt:.2f}ms; tok/s: {tokens_per_sec:.02f}")