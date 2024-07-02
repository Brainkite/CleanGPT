import os
import torch
import numpy as np

def load_tokens(filename):
    toks = torch.tensor(np.load(filename), dtype=torch.long)
    return toks

class DistributedDataloader:
    def __init__(self, data_dir, B, T, process_rank=0, num_processes=1, split='train', shuffle=False):
        self.data_dir = data_dir
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.shuffle = shuffle
        assert split in {'train', 'val'}
        
        self.load_dataset()
        self.reset()
        
    def reset(self):
        self.curr_position = 0
        
        
    def load_dataset(self):
        fns = os.listdir(self.data_dir)
        fns = sorted([fn for fn in fns if self.split in fn])
        fns = [os.path.join(self.data_dir, fn) for fn in fns]
        assert len(fns)>0
        if self.shuffle:
            print("Shuffling dataset")
            shuffled_idxs = torch.randperm(len(fns))
            fns = [fns[i] for i in shuffled_idxs]
        
        if self.split == 'train':
            # Split dataset by processes
            split_size = len(fns)//self.num_processes
            split_start = split_size * self.process_rank
            split_end = split_start + split_size
            split = fns[split_start : split_end]
            self.shards = split
        else:
            self.shards = fns

        print(f"Found {len(self.shards)} shards for {self.split} split in process {self.process_rank}")
        
        print('### Loading shards in one tensor...')
        self.total_samples = sum(load_tokens(shard).size(0) // (self.T + 1) for shard in self.shards)
        self.tokens = torch.empty((self.total_samples, self.T + 1), dtype=torch.long)

        current_index = 0
        for shard in self.shards:
            toks = load_tokens(shard)
            n_samples = toks.size(0) // (self.T + 1)
            toks = toks[:n_samples * (self.T + 1)].view(n_samples, self.T + 1)
            self.tokens[current_index:current_index + n_samples, :] = toks
            current_index += n_samples

        if self.shuffle:
            self.tokens = self.tokens[torch.randperm(self.tokens.size(0))]

        print(f"Loaded {self.tokens.shape} samples in process {self.process_rank}")
    
    def new_batch(self):
        B = self.B
        
        if self.curr_position + B <= self.tokens.size(0):
            buf = self.tokens[self.curr_position : self.curr_position + B]
            self.curr_position += B
        else:
            remain = self.tokens.size(0) - self.curr_position
            buf = torch.empty((B, self.T+1), dtype=torch.long)
            buf[:remain] = self.tokens[self.curr_position:]
            buf[remain:] = self.tokens[:B-remain]
            self.curr_position = B - remain

        x = buf[:, :-1]
        y = buf[:, 1:]
        
        assert x.size(1) == self.T, f"Expected x shape (B, {self.T}), got {x.shape}"
        assert y.size(1) == self.T, f"Expected y shape (B, {self.T}), got {y.shape}"
        return x, y
        
        
