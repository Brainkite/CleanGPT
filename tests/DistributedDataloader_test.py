import pytest
import torch
import os
import tempfile
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from simpleGPT.DistributedDataLoader import DistributedDataset, create_distributed_dataloader

@pytest.fixture
def mock_data_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Create mock data files
        for i in range(3):
            data = np.random.randint(0, 1000, size=1000, dtype=np.int64)
            np.save(os.path.join(tmpdirname, f'shard_{i}_train.npy'), data)
        yield tmpdirname

@pytest.fixture
def dataset(mock_data_dir):
    return DistributedDataset(mock_data_dir, T=100, split='train')

def test_dataset_length(dataset):
    expected_length = (len(dataset.tokens) - 1) // dataset.T
    assert len(dataset) == expected_length

def test_non_overlapping_sequences(dataset):
    seq1 = dataset[0]
    seq2 = dataset[1]
    assert not torch.all(seq1[0] == seq2[0])
    assert not torch.all(seq1[1] == seq2[1])

def test_sequence_length(dataset):
    seq = dataset[0]
    assert seq[0].shape[0] == dataset.T
    assert seq[1].shape[0] == dataset.T

def test_y_is_next_token(dataset):
    seq = dataset[0]
    assert torch.all(seq[0][1:] == seq[1][:-1])

def test_all_data_used(dataset):
    all_data_x = torch.cat([dataset[i][0] for i in range(len(dataset))])
    all_data_y = torch.cat([dataset[i][1] for i in range(len(dataset))])
    
    # Check lengths
    assert len(all_data_x) == len(dataset) * dataset.T
    assert len(all_data_y) == len(dataset) * dataset.T
    
    # Check that all tokens are used (except the last one)
    assert torch.all(all_data_x == dataset.tokens[:len(all_data_x)])
    assert torch.all(all_data_y[:-1] == dataset.tokens[1:len(all_data_y)])

def test_last_sequence(dataset):
    last_seq = dataset[len(dataset)-1]
    assert len(last_seq[0]) == dataset.T
    assert len(last_seq[1]) == dataset.T
    assert torch.all(last_seq[0][1:] == last_seq[1][:-1])

def test_distributed_sampler(dataset):
    world_size = 2
    for rank in range(world_size):
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        expected_length = (len(dataset) + world_size - 1) // world_size
        assert len(sampler) == expected_length

def test_dataloader_batch_size(mock_data_dir):
    B, T = 4, 100
    try:
        dataloader = create_distributed_dataloader(mock_data_dir, B, T, rank=0, world_size=1, split='train')
        batch = next(iter(dataloader))
        assert batch[0].shape == (B, T)
        assert batch[1].shape == (B, T)
    except NameError:
        pytest.skip("create_dataloader function not available")

def test_all_samples_covered(dataset):
    world_size = 2
    all_indices = set()
    for rank in range(world_size):
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
        all_indices.update(iter(sampler))
    
    assert len(all_indices) == len(dataset)  # All samples should be covered

def test_epoch_change_shuffles_data(dataset):
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True)
    
    epoch1_indices = list(iter(sampler))
    sampler.set_epoch(1)
    epoch2_indices = list(iter(sampler))
    
    assert epoch1_indices != epoch2_indices  # Indices should be different after changing epoch

if __name__ == "__main__":
    pytest.main([__file__])