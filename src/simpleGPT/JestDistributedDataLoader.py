import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import time

def load_tokens(filename):
    toks = torch.tensor(np.load(filename), dtype=torch.long)
    return toks

def load_ref_scores(filename):
    scores = torch.tensor(np.load(filename), dtype=torch.float32)
    return scores

class JestDistributedDataloader:
    def __init__(self, data_dir, B, T, process_rank=0, num_processes=1, split='train', shuffle=False, jest=False, ref_scores_fp=None):
        self.data_dir = data_dir
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.shuffle = shuffle
        self.jest=jest
        self.ref_scores_fp = str(ref_scores_fp)
        
        assert split in {'train', 'val'}
        
        self.load_dataset()
        
        if self.jest: 
            self.load_ref_scores()
            
        if self.shuffle:
            self.shuffle_samples()
            
        self.curr_position = 0   
    
    def __len__(self):
        return 1 + self.samples.shape[0] // self.B 
        
    def load_dataset(self):
        fns = os.listdir(self.data_dir)
        fns = sorted([fn for fn in fns if self.split in fn])
        fns = [os.path.join(self.data_dir, fn) for fn in fns]
        self.shards = fns
        assert len(fns)>0
        
        self.shards_idxs = np.arange(len(fns))
        if self.shuffle:
            print("# Shuffling dataset before rank split")
            np.random.shuffle(self.shards_idxs)
            
        if self.split == 'train':
            # Split dataset by processes
            self.shards_idxs = list(np.array_split(self.shards_idxs, self.num_processes)[self.process_rank])

        print(f"# Found {len(self.shards_idxs)} shards for {self.split} split in process {self.process_rank}")
        
        self.total_samples = sum(load_tokens(self.shards[i]).size(0) // (self.T + 1) for i in self.shards_idxs)
        print(f'# Loading {self.total_samples} samples in one tensor...')
        self.samples = torch.empty((self.total_samples, self.T + 1), dtype= torch.long)

        current_index = 0
        for shard_i in self.shards_idxs:
            toks = load_tokens(self.shards[shard_i])
            n_samples = toks.size(0) // (self.T + 1)
            toks = toks[:n_samples * (self.T + 1)].view(n_samples, self.T + 1)
            self.samples[current_index:current_index + n_samples, :] = toks
            current_index += n_samples

        print(f"# Loaded {self.samples.shape} samples in process {self.process_rank}")
        
    def load_ref_scores(self):
        dtype = np.dtype([('shard_idx', np.uint16), ('sample_idx', np.uint32), ('score', np.float32)])
        sample_idx_scores = np.zeros(self.total_samples, dtype=dtype)

        current_index = 0
        for shard_i in self.shards_idxs:
            n_samples = load_tokens(self.shards[shard_i]).size(0) // (self.T + 1)
            end_index = current_index + n_samples
            sample_idx_scores['shard_idx'][current_index:end_index] = shard_i
            sample_idx_scores['sample_idx'][current_index:end_index] = np.arange(n_samples)
            current_index += n_samples
        
        if os.path.exists(self.ref_scores_fp):
            self.ref_scores = np.load(self.ref_scores_fp)
            matching_row_idxs = np.where(
                np.in1d(
                    self.ref_scores[['shard_idx', 'sample_idx']], 
                    sample_idx_scores[['shard_idx', 'sample_idx']], 
                    assume_unique=True))[0]
            self.ref_scores = self.ref_scores[matching_row_idxs]
        else:
            parent_dir = os.path.dirname(self.ref_scores_fp)
            os.makedirs(parent_dir, exist_ok=True)
            self.ref_scores = sample_idx_scores
        
        assert self.ref_scores.shape[0] == self.samples.shape[0], print(self.ref_scores.shape, self.samples.shape)
        
    def shuffle_samples(self):
        shuffle_idxs = torch.randperm(self.samples.size(0))
        self.samples = self.samples[shuffle_idxs, :]
        if self.jest:
            self.ref_scores = self.ref_scores[shuffle_idxs]
    
    def save_ref_scores(self):
        sort_idxs = np.lexsort((self.ref_scores['sample_idx'], self.ref_scores['shard_idx']))
        sorted_ref_scores = self.ref_scores[sort_idxs]
        print(f"Writting ref scores to {self.ref_scores_fp}")
        np.save(self.ref_scores_fp, sorted_ref_scores)

    def new_batch(self):
        B = self.B
        ref_sc = None
        
        if self.curr_position + B <= self.samples.size(0):
            buf = self.samples[self.curr_position : self.curr_position + B, :]
            if self.jest:
                ref_sc = self.ref_scores[self.curr_position : self.curr_position + B]
            self.curr_position += B
        else:
            remain = self.samples.size(0) - self.curr_position
            buf = torch.empty((B, self.T+1), dtype=torch.long)
            buf[:remain] = self.samples[self.curr_position:]
            buf[remain:] = self.samples[:B-remain]
            
            if self.jest:
                ref_sc = np.concatenate([
                    self.ref_scores[self.curr_position:], 
                    self.ref_scores[:B-remain]
                    ],axis=0)
            
            self.curr_position = B - remain

        x = buf[:, :-1]
        y = buf[:, 1:]

        return x, y, ref_sc
        
    def compute_and_cache_ref_scores(self, model_name, device, device_type, autocast_bf16, compile_model):
        assert model_name in {
            'EleutherAI/gpt-j-6b', 
            'openai-community/gpt2',
            'openai-community/gpt2-large',
            'openai-community/gpt2-medium',
            'openai-community/gpt2-xl',
            'EleutherAI/gpt-neo-125m',
            'EleutherAI/gpt-neo-1.3B',
            'EleutherAI/gpt-neo-2.7B'
            }
        print(f"# Loading Hugging Face model: {model_name}")
        
        def _forward_loss(x, y):
            outputs = model(x, labels=y)
            loss = outputs.loss.item()

            # Compute per-sample loss
            probs = outputs.logits.view(-1, model.config.vocab_size)
            labs = y.flatten()
            scores = torch.nn.functional.cross_entropy(
                probs,
                labs,
                reduction='none'
            )
            scores = scores.view(y.size())
            return loss, scores
        
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        
        if compile_model:
            print('# Compiling model')
            model = torch.compile(model)
        model.eval()
        total_loss = 0        
        total_batches = 1 + self.samples.shape[0] // self.B
        if self.process_rank == 0:
            print(f'# Total batches to compute: {total_batches}')
            
        dtype = np.dtype([('shard_idx', np.uint16), ('sample_idx', np.uint32), ('score', np.float32)])
        new_ref_scores = np.zeros((self.samples.shape[0],), dtype=dtype)
        total_tokens_processed = 0

        with torch.no_grad():
            
            if self.process_rank == 0:
                pbar = tqdm(total=total_batches, desc="Computing reference scores")
                start_time = time.time()
            
            curr_position = 0
            for i in range(total_batches):
                x, y, ref_sc = self.new_batch()
                x = x.to(device)
                y = y.to(device)

                if autocast_bf16:
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        loss, scores = _forward_loss(x,y)
                else:
                    loss, scores = _forward_loss(x,y)
                    
                # Average loss per sample
                scores = scores.mean(dim=1).cpu().numpy()
                ref_sc['score'] = scores
                
                start = curr_position
                end = curr_position + self.B 
                if end > self.samples.size(0):
                    remain = self.samples.size(0) - end
                    end = self.samples.size(0)
                    new_ref_scores[start:] = ref_sc[:remain]
                else:
                    new_ref_scores[start: end] = ref_sc

                total_loss += loss * len(scores)
                total_samples = i * self.B
                total_tokens_processed = total_samples * self.T

                if self.process_rank == 0:
                    elapsed_time = time.time() - start_time
                    tokens_per_second = total_tokens_processed / elapsed_time
                    batches_per_second = (i + 1) / elapsed_time
                    eta_seconds = (total_batches - (i + 1)) / batches_per_second
                    eta_string = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                    pbar.set_postfix({
                        'Batch': f'{i+1}/{total_batches}',
                        'Avg Loss': f'{total_loss/total_samples:.4f}',
                        'Tokens/sec': f'{tokens_per_second:.2f}',
                        'ETA': eta_string
                    })
                    pbar.update(1)

            if self.process_rank == 0:
                pbar.close()

        average_loss = total_loss / total_samples
        if self.process_rank == 0:
            print(f"# Finished computing and caching reference scores. Average loss: {average_loss:.4f}")

        self.ref_scores = new_ref_scores
        self.save_ref_scores()
