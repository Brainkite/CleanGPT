import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def load_tokens(filename):
    toks = torch.tensor(np.load(filename), dtype=torch.long)
    return toks

def load_ref_scores(filename):
    scores = torch.tensor(np.load(filename), dtype=torch.float32)
    return scores

class JestDistributedDataloader:
    def __init__(self, data_dir, B, T, process_rank=0, num_processes=1, split='train', shuffle=False, jest=False, ref_scores_dir=None):
        self.data_dir = data_dir
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        self.shuffle = shuffle
        self.jest=jest
        self.ref_scores_dir = str(ref_scores_dir)
        assert split in {'train', 'val'}
        
        self.load_dataset()
        
        if self.jest: 
            self.load_ref_scores()
            
        if self.shuffle:
            self.shuffle_samples()
            
        self.curr_position = 0   
    
    def __len__(self):
        return 1 + self.tokens.shape[0] // self.B 
        
    def load_dataset(self):
        fns = os.listdir(self.data_dir)
        fns = sorted([fn for fn in fns if self.split in fn])
        fns = [os.path.join(self.data_dir, fn) for fn in fns]
        assert len(fns)>0
        if self.shuffle:
            print("# Shuffling dataset before rank split")
            shuffled_idxs = torch.randperm(len(fns))
            fns = [fns[i] for i in shuffled_idxs]
            
        if self.split == 'train':
            # Split dataset by processes
            self.shards = list(np.array_split(fns, self.num_processes)[self.process_rank])
        else:
            self.shards = fns

        print(f"# Found {len(self.shards)} shards for {self.split} split in process {self.process_rank}")
        
        print('# Loading shards in one tensor...')
        self.total_samples = sum(load_tokens(shard).size(0) // (self.T + 1) for shard in self.shards)
        self.tokens = torch.empty((self.total_samples, self.T + 1), dtype= torch.long)

        current_index = 0
        for shard in self.shards:
            toks = load_tokens(shard)
            n_samples = toks.size(0) // (self.T + 1)
            toks = toks[:n_samples * (self.T + 1)].view(n_samples, self.T + 1)
            self.tokens[current_index:current_index + n_samples, :] = toks
            current_index += n_samples

        print(f"# Loaded {self.tokens.shape} samples in process {self.process_rank}")
        
    def load_ref_scores(self):
        dtype = np.dtype([('shard_idx', np.uint16), ('sample_idx', np.uint16), ('score', np.float32)])
        self.ref_scores = np.zeros(self.total_samples, dtype=dtype)
        
        if not os.path.exists(self.ref_scores_dir): os.makedirs(self.ref_scores_dir)

        current_index = 0
        for shard_i, shard in enumerate(self.shards):
            n_samples = load_tokens(shard).size(0) // (self.T + 1)
            
            ref_score_pth = os.path.join(self.ref_scores_dir, os.path.basename(shard))
            if not os.path.exists(ref_score_pth):
                print(f'# Ref score file absent, creating one with zeros at {ref_score_pth}')
                scores = np.zeros((n_samples,), dtype=np.float32)
                scores = torch.tensor(scores, dtype=torch.float32)
            else:
                scores = load_ref_scores(ref_score_pth)
                assert n_samples == scores.size(0), print(n_samples, "!=", scores.size(0))
            
            end_index = current_index + n_samples
            self.ref_scores['shard_idx'][current_index:end_index] = shard_i
            self.ref_scores['sample_idx'][current_index:end_index] = np.arange(n_samples)
            self.ref_scores['score'][current_index:end_index] = scores
            
            current_index += n_samples

        
    def shuffle_samples(self):
        shuffle_idxs = torch.randperm(self.tokens.size(0))
        self.tokens = self.tokens[shuffle_idxs, :]
        if self.jest:
            self.ref_scores = self.ref_scores[shuffle_idxs]
            self.shard_sample_idxs = self.shard_sample_idxs[shuffle_idxs, :]
    
    def save_ref_scores(self):
        sort_idxs = np.lexsort((self.ref_scores['sample_idx'], self.ref_scores['shard_idx']))
        sorted_ref_scores = self.ref_scores[sort_idxs]
        
        shard_idxs, inverse_indices = np.unique(sorted_ref_scores['shard_idx'], return_inverse=True)
        shards_ref_scores = {os.path.basename(self.shards[shard_id]): sorted_ref_scores[inverse_indices == i] for i, shard_id in enumerate(shard_idxs)}
        
        for shard, ref_scores in shards_ref_scores.items():
            fn = os.path.join(self.ref_scores_dir, shard)
            np.save(fn, ref_scores['score'].astype(np.float32)) 
            
    def update_ref_scores(self, ref_sc):
        # Find the row indices in ref_scores that match ref_sc rows
        matching_row_idxs = np.where(
            np.in1d(
                self.ref_scores[['shard_idx', 'sample_idx']], 
                ref_sc[['shard_idx', 'sample_idx']], 
                assume_unique=True)
            )[0]
        
        # Update the 'score' column of ref_scores at these indices
        self.ref_scores['score'][matching_row_idxs] = ref_sc['score']

      
    def new_batch(self):
        B = self.B
        ref_sc = None
        
        if self.curr_position + B <= self.tokens.size(0):
            buf = self.tokens[self.curr_position : self.curr_position + B, :]
            if self.jest:
                ref_sc = self.ref_scores[self.curr_position : self.curr_position + B]
            self.curr_position += B
        else:
            remain = self.tokens.size(0) - self.curr_position
            buf = torch.empty((B, self.T+1), dtype=torch.long)
            buf[:remain] = self.tokens[self.curr_position:]
            buf[remain:] = self.tokens[:B-remain]
            
            if self.jest:
                ref_sc = np.concatenate([
                    self.ref_scores[self.curr_position:], 
                    self.ref_scores[:B-remain]
                    ],axis=0)
            
            self.curr_position = B - remain

        x = buf[:, :-1]
        y = buf[:, 1:]

        return x, y, ref_sc
        
    def compute_and_cache_ref_scores(self, model_name, device):
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
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        # We don't need to initialize the tokenizer since input is pre-tokenized

        model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for i in tqdm(range(1 + self.tokens.shape[0] // self.B ), desc="Computing reference scores"):
                x, y, ref_sc = self.new_batch()
                x = x.to(device)
                y = y.to(device)

                # The input is already tokenized, so we can use it directly
                outputs = model(x, labels=y)
                loss = outputs.loss.item()

                # Compute per-sample loss
                # Note: we use the model's config.vocab_size instead of output.logits.size(-1)
                probs = outputs.logits.view(-1, model.config.vocab_size)
                labs = y.flatten()
                scores = torch.nn.functional.cross_entropy(
                    probs,
                    labs,
                    reduction='none'
                )
                scores = scores.view(y.size())

                # Average loss per sample
                scores = scores.mean(dim=1).cpu().numpy()

                # Update ref_sc with new scores
                ref_sc['score'] = scores

                # Update the reference scores in the main array
                self.update_ref_scores(ref_sc)

                total_loss += loss * len(scores)
                total_samples += len(scores)

        average_loss = total_loss / total_samples
        print(f"# Finished computing and caching reference scores. Average loss: {average_loss:.4f}")

        self.save_ref_scores()  
