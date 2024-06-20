from dataclasses import dataclass
import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    use_flash_attn: bool = True

class CasualSelfAttention(nn.Module):
    """
    Perform a batched multi-head attention
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.use_flash_attn = config.use_flash_attn
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        setattr(self.c_proj, 'GPT_SCALE_INIT', 1.)
        
        if not self.use_flash_attn:
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

        # Organize q,k,v by att heads
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (bs, nh, seq, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (bs, nh, seq, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (bs, nh, seq, hs)

        if self.use_flash_attn:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # (bs, nh) are in batch dims so we can batch compute att scores per head
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (bs, nh, seq, seq)
            # Discard upper triangular att scores
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (bs, nh, seq, hs)
        
        # Catenate all heads (bs, nh, seq, hs)-> (bs, seq, n_embd)
        y = y.transpose(1,2).contiguous().view(B,T,C)

        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        setattr(self.c_proj, 'GPT_SCALE_INIT', 1.)

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
        
        # Weight sharing scheme of "Attention is all you need"
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        # This method tries to reproduce the initialization method of original GPT2 paper.
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'GPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # 2*n_layers because each block has 2 residual connexions
            torch.nn.init.normal_(module.weight, mean=0., std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0., std=0.02)
    
    def forward(self, idx, targets=None):
        B,T = idx.size()
        assert T <= self.config.block_size, f"{T}, {self.config.block_size}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        tok_emb = self.transformer.wte(idx) # (B, T, n_emb)
        
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), # (B*T, vocab_size)
                targets.view(-1) # (B*T,)
                )
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn,p in self.named_parameters() if p.requires_grad}
        # Apply wd only to matmul layers (linear, embedding), not to biases and layernorm.
        decay_params = [p for n,p in param_dict.items() if p.dim()>=2]
        nodecay_params = [p for n,p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"Num decayed param tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"Num non-decayed param tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        
        fused_avilable = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_avilable and 'cuda' in device
        print(f"Using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
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
        

        


        
        
