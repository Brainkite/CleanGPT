import pytest
import numpy as np
import torch
import os
from unittest.mock import patch, MagicMock
from simpleGPT.JestDistributedDataLoader import JestDistributedDataloader, load_tokens, load_ref_scores

@pytest.fixture
def sample_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(3):
        np.save(data_dir / f"train_shard_{i}.npy", np.random.randint(0, 50257, size=(1040,)))
    return str(data_dir)

@pytest.fixture
def sample_ref_scores_dir(tmp_path):
    ref_scores_dir = tmp_path / "ref_scores"
    ref_scores_dir.mkdir()
    return str(ref_scores_dir)

def test_init_not_jest(sample_data_dir, sample_ref_scores_dir):
    loader = JestDistributedDataloader(sample_data_dir, 32, 128, jest=False, ref_scores_dir=sample_ref_scores_dir)
    assert loader.B == 32
    assert loader.T == 128
    assert loader.jest == False
    assert loader.ref_scores_dir == sample_ref_scores_dir

def test_init_with_jest(sample_data_dir, sample_ref_scores_dir):
    loader = JestDistributedDataloader(sample_data_dir, 32, 128, jest=True, ref_scores_dir=sample_ref_scores_dir)
    assert loader.B == 32
    assert loader.T == 128
    assert loader.jest == True
    assert loader.ref_scores_dir == sample_ref_scores_dir

def test_load_dataset(sample_data_dir):
    loader = JestDistributedDataloader(sample_data_dir, 32, 128)
    assert len(loader.shards) == 3
    assert loader.tokens.shape[0] == 24
    assert loader.tokens.shape[1] == 129  # T + 1

def test_loading_ref_scores(sample_data_dir, sample_ref_scores_dir):
    loader = JestDistributedDataloader(sample_data_dir, 32, 128, jest=True, ref_scores_dir=sample_ref_scores_dir)
    assert loader.ref_scores.shape[0] == loader.total_samples
    assert 'shard_idx' in loader.ref_scores.dtype.names
    assert 'sample_idx' in loader.ref_scores.dtype.names
    assert 'score' in loader.ref_scores.dtype.names

def test_shuffle_samples(sample_data_dir):
    loader = JestDistributedDataloader(sample_data_dir, 32, 128, shuffle=False)
    original_tokens = loader.tokens.clone()
    loader.shuffle_samples()
    assert not torch.all(loader.tokens == original_tokens)

def test_save_ref_scores(sample_data_dir, sample_ref_scores_dir):
    loader = JestDistributedDataloader(sample_data_dir, 32, 128, jest=True, ref_scores_dir=sample_ref_scores_dir)
    loader.save_ref_scores()
    assert len(os.listdir(sample_ref_scores_dir)) == len(loader.shards)

def test_update_ref_scores(sample_data_dir, sample_ref_scores_dir):
    loader = JestDistributedDataloader(sample_data_dir, 32, 128, jest=True, ref_scores_dir=sample_ref_scores_dir, shuffle=False)
    new_scores = np.ones(4, dtype=loader.ref_scores.dtype)
    new_scores['shard_idx'] = 0
    new_scores['sample_idx'] = np.arange(4)
    new_scores['score'] = np.random.rand(4)
    loader.update_ref_scores(new_scores)
    assert np.all(loader.ref_scores[:4]['score'] == new_scores['score'])

def test_new_batch_normal(sample_data_dir):
    loader = JestDistributedDataloader(sample_data_dir, 32, 128)
    x, y, ref_sc = loader.new_batch()
    assert x.shape == (32, 128)
    assert y.shape == (32, 128)
    assert ref_sc is None  # because jest is False

def test_new_batch_jest(sample_data_dir, sample_ref_scores_dir):
    loader = JestDistributedDataloader(sample_data_dir, 32, 128, jest=True, ref_scores_dir=sample_ref_scores_dir)
    x, y, ref_sc = loader.new_batch()
    assert x.shape == (32, 128)
    assert y.shape == (32, 128)
    assert ref_sc.shape[0] == 32

def test_new_batch_wrap_around(sample_data_dir):
    loader = JestDistributedDataloader(sample_data_dir, 32, 128)
    # Move to the end of the dataset
    loader.curr_position = loader.tokens.shape[0] - 16
    x, y, ref_sc = loader.new_batch()
    assert x.shape == (32, 128)
    assert y.shape == (32, 128)
    assert loader.curr_position == 16

def test_load_tokens():
    # Create a temporary file with known content
    data = np.array([1, 2, 3, 4, 5])
    np.save('temp_tokens.npy', data)
    
    # Load the tokens
    loaded_tokens = load_tokens('temp_tokens.npy')
    
    # Check if the loaded tokens match the original data
    assert torch.all(loaded_tokens == torch.tensor(data))
    
    # Clean up
    os.remove('temp_tokens.npy')

def test_load_ref_scores():
    # Create a temporary file with known content
    data = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    np.save('temp_scores.npy', data)
    
    # Load the scores
    loaded_scores = load_ref_scores('temp_scores.npy')
    
    # Check if the loaded scores match the original data
    assert torch.all(torch.isclose(loaded_scores, torch.tensor(data)))
    
    # Clean up
    os.remove('temp_scores.npy')

def test_compute_and_cache_ref_scores(sample_data_dir, sample_ref_scores_dir):
    loader = JestDistributedDataloader(
        sample_data_dir, 2, 128, 
        jest=True, ref_scores_dir=sample_ref_scores_dir, 
        shuffle=False)
    original_scores = loader.ref_scores.copy()
    
    loader.compute_and_cache_ref_scores('openai-community/gpt2', 'cpu')
    
    assert np.all(loader.ref_scores['score'] != original_scores['score'])
    
    
    
    