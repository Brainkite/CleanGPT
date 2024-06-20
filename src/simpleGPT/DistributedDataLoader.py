import os
import torch
import numpy as np

def load_tokens(filename):
    toks = torch.tensor(np.load(filename).astype(np.int64), dtype=torch.long)
    return toks

class DistributedDataloader:
    def __init__(self, data_dir, B, T, process_rank=0, num_processes=1, split='train'):
        self.data_dir = data_dir
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        assert split in {'train', 'val'}
        
        self.load_dataset()
        
        # Init position
        self.curr_shard = 0
        self.tokens = load_tokens(self.shards[self.curr_shard])
        self.curr_position = self.B * self.T * self.process_rank
        
    def load_dataset(self):
        fns = os.listdir(self.data_dir)
        fns = sorted([fn for fn in fns if self.split in fn])
        fns = [os.path.join(self.data_dir, fn) for fn in fns]
        self.shards = fns
        assert len(fns)>0
        if self.process_rank == 0:
            print(f"Found {len(fns)} shards for {self.split} split")
    
    def new_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.curr_position : self.curr_position + B*T+1]
        x = buf[:-1].view(B,T)
        y = buf[1:].view(B,T)
        
        self.curr_position += B * T * self.num_processes
        
        if self.curr_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.curr_shard = (self.curr_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.curr_shard])
            self.curr_position = self.B * self.T * self.process_rank
        return x, y
        
        