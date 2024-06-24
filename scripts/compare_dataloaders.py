import torch

from simpleGPT.DistributedDataLoader import DistributedDataloader
from karpathy_gpt import DataLoaderLite

def test_dataloader_implementations(data_dir, B, T, process_rank, num_processes, split, num_batches):
    # Initialize both dataloaders
    dl1 = DistributedDataloader(data_dir, B, T, process_rank, num_processes, split)
    dl2 = DataLoaderLite(B, T, process_rank, num_processes, split)

    # Check if both dataloaders yield the same outputs
    for i in range(num_batches):
        x1, y1 = dl1.new_batch()
        x2, y2 = dl2.next_batch()

        assert torch.equal(x1, x2), f"Mismatch in x at batch {i}"
        assert torch.equal(y1, y2), f"Mismatch in y at batch {i}"

    print(f"All {num_batches} batches matched successfully.")

# Example usage
data_dir = '/workspace/datasets/edu_fineweb10B'  # Adjust according to your dataset path
B = 2  # Batch size
T = 64  # Sequence length
process_rank = 0  # Process rank for distributed loading
num_processes = 1  # Total number of processes
split = 'train'  # Data split
num_batches = 1000000  # Number of batches to compare

test_dataloader_implementations(data_dir, B, T, process_rank, num_processes, split, num_batches)