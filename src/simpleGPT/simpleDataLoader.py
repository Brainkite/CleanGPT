import tiktoken
import torch

class SimpleDataloader:
    def __init__(self, fn, B, T):
        self.B = B
        self.T = T
        
        enc = tiktoken.get_encoding('gpt2')
        with open(fn, 'r') as f:
            tokens = enc.encode(f.read())
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens // (B*T))} batches")
        
        self.current_position = 0
    
    def new_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T+1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        
        self.current_position += B*T
        
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        return x, y
        