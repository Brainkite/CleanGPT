from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CasualSelfAttention(nn.Module):
    """
    Perform a batched multi-head attention
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Create attention lower-triangular mask to keep only attn score with previous tokens
        self.register_buffer("bias",
                             torch.tril(
                                 torch.ones(config.block_size, config.block_size)
                             ).view(1,1,config.block_size, config.block_size)
                             )
    
    def forward(self, x):
        B,T,C = x.size() # batch_size, seg_length, n_embeddings

        #Compute batched multihead Q,K,V
        qkv = self.c_attn(x) # (bs, seq, 3*n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Organize q,k,v by att heads
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (bs, nh, seq, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (bs, nh, seq, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (bs, nh, seq, hs)

        # (bs, nh) are in batch dims so we can batch compute att scores per head
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (bs, nh, seq, seq)

        # Discard upper triangular att scores
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (bs, nh, seq, hs)
        # Catenate all heads (bs, nh, seq, hs)-> (bs, seq, n_embd)
        y = y.transpose(1,2).contiguous().view(B,T,C)

        return self.c_proj(y)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                h = nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),
                ln_f = nn.LayerNorm(config.n_embd)
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
    
    def forward(self, idx):
        B,T = idx.size()
        assert T <= self.config.block_size
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_emb)
        x = tok_emb + pos_emb
        
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits
        
    
    @classmethod
    def from_pretrained(cls, model_type: str):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        print(f"Loading weights from {model_type}")
        from transformers import GPT2LMHeadModel
        
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768), #124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), #350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280), #774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600) #1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        
        config = GPTConfig(**config_args)
        model = GPT(config)
        
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] #mask, not a param
        
        # init a HF model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        to_transpose = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys_hf) == len(sd_keys), f"failed to match keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in to_transpose):
                assert sd_hf[k].shape[::-1] == sd[k].shape, f"Shape of key {k} do not match: {sd_hf[k].shape[::-1]} != {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape, f"Shape of key {k} do not match: {sd_hf[k].shape} != {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model
        
#-----------------------------------------------------------------------------------------
        
num_return_seq = 5
max_length = 30

model = GPT.from_pretrained('gpt2')
model.eval()
model.to('cuda')

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_seq, 1)
x = tokens.to('cuda')

torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) # (B,T,vocab_size)
        logits = logits[:, -1, :] # (B,vocab_size)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) # (B,50) (B,50)
        ix = torch.multinomial(topk_probs, 1) #idxs of top prob (B,1)
        xcol = torch.gather(topk_indices, -1, ix) # vocab idxs from selected top prob (B,1)
        x = torch.cat((x, xcol), dim=1) # (B, T+1)
    
    
for i in range(num_return_seq):
    tokens = x[i, :max_length].tolist()
    decode = enc.decode(tokens)
    print(decode)
        


        
        
