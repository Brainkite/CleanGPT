import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

def load_tokens(filename):
    return torch.tensor(np.load(filename), dtype=torch.long)

class DistributedDataset(Dataset):
    def __init__(self, data_dir, T, split='train'):
        self.data_dir = data_dir
        self.T = T
        self.split = split
        assert split in {'train', 'val'}
        
        self.load_dataset()
        
    def load_dataset(self):
        fns = os.listdir(self.data_dir)
        fns = sorted([fn for fn in fns if self.split in fn])
        fns = [os.path.join(self.data_dir, fn) for fn in fns]
        assert len(fns) > 0
        print(f"Found {len(fns)} shards for {self.split} split")
        
        # Pre-load all shards into RAM
        print('### Loading shards in one tensor...')
        self.tokens = torch.cat([load_tokens(shard) for shard in fns])
        print(f"Loaded {len(self.tokens)} tokens into RAM")
        
        # Calculate the number of non-overlapping sequences
        self.num_samples = len(self.tokens) // self.T

    def __getitem__(self, idx):
        start_idx = idx * self.T
        end_idx = start_idx + self.T
        x = self.tokens[start_idx : end_idx]
        y = self.tokens[start_idx + 1 : end_idx + 1]
        return x, y

    def __len__(self):
        return (len(self.tokens) - 1) // self.T

def create_distributed_dataloader(data_dir, B, T, rank, world_size, split='train'):
    dataset = DistributedDataset(data_dir, T, split)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=(split == 'train'))
    
    dataloader = DataLoader(
        dataset,
        batch_size = B,
        sampler = sampler,
        num_workers = int(os.getenv('NGPUS')) * 6,
        pin_memory = True,
        prefetch_factor = 4,
        persistent_workers = True
    )
    
    return dataloader


