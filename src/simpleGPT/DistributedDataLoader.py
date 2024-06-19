import tiktoken
import torch

class DistributedDataloader:
    def __init__(self, fn, B, T, process_rank=0, num_processes=1):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        
        enc = tiktoken.get_encoding('gpt2')
        with open(fn, 'r') as f:
            tokens = enc.encode(f.read())
        self.tokens = torch.tensor(tokens)
        if process_rank == 0:
            print(f"Loaded {len(self.tokens)} tokens")
            print(f"1 epoch = {len(self.tokens // (B*T))} batches")
        
        self.current_position = self.B * self.T * self.process_rank
    
    def new_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T+1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        
        self.current_position += B * T * self.num_processes
        
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = 0
        return x, y
        