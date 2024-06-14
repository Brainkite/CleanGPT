import torch
import torch.functional as F
from simpleGPT.simplegpt import GPT, GPTConfig
from simpleGPT.simpleDataLoader import SimpleDataloader
import sys
import time
import gc

num_return_seq = 5
max_length = 30
B,T = 8, 1024
device = 'cuda'

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

def train():
    train_loader = SimpleDataloader("dataset/tiny_shakesprear.txt", B, T)
    model = GPT(GPTConfig())
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
    for i in range(100):
        t0 = time.time()
        x, y = train_loader.new_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)*1000
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1-t0)
        print(f"Step {i}: loss: {loss.item()}n dt: {dt:.2f}ms, tok/s: {tokens_per_sec:.02f}")



print('\n##########  Default precision')
train()
gc.collect()

print('\n##########  TF32 matmul precision')
torch.set_float32_matmul_precision('high')
train()
gc.collect()

print('\n##########  BF16 matmul precision')
torch.set_float32_matmul_precision('medium')
train()
gc.collect()


def train_bf16():
    train_loader = SimpleDataloader("dataset/tiny_shakesprear.txt", B, T)
    model = GPT(GPTConfig())
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4)
    for i in range(100):
        t0 = time.time()
        x, y = train_loader.new_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16)
            logits, loss = model(x, y)
            loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1-t0)*1000
        tokens_per_sec = (train_loader.B * train_loader.T) / (t1-t0)
        print(f"Step {i}: loss: {loss.item()}n dt: {dt:.2f}ms, tok/s: {tokens_per_sec:.02f}")

print('\n##########  Default precision, autocast bf16')
train_bf16()
gc.collect()

print('\n##########  TF32 matmul precision, autocast bf16')
torch.set_float32_matmul_precision('high')
train_bf16()
gc.collect()

print('\n##########  BF16 matmul precision, autocast bf16')
torch.set_float32_matmul_precision('medium')
train_bf16()
gc.collect()

sys.exit(0)