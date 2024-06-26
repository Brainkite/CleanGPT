import pytest
import torch
from simpleGPT.SimpleDataLoader import SimpleDataloader

class TestSimpleDataloader:
    def test_initialization_with_valid_file(self):
        fn = 'valid_file.txt'
        B, T = 2, 3
        with open(fn, 'w') as f:
            f.write("This is a test file for SimpleDataloader.")
        dataloader = SimpleDataloader(fn, B, T)
        assert dataloader.B == B
        assert dataloader.T == T
        assert len(dataloader.tokens) > 0

    def test_proper_loading_and_tokenization(self):
        fn = 'valid_file.txt'
        B, T = 2, 3
        with open(fn, 'w') as f:
            f.write("This is a test file for SimpleDataloader.")
        dataloader = SimpleDataloader(fn, B, T)
        assert len(dataloader.tokens) > 0

    def test_generate_batches_valid_B_T(self):
        fn = 'valid_file.txt'
        B, T = 2, 3
        with open(fn, 'w') as f:
            f.write("This is a test file for SimpleDataloader.")
        dataloader = SimpleDataloader(fn, B, T)
        x, y = dataloader.new_batch()
        assert x.shape == torch.Size([B, T])
        assert y.shape == torch.Size([B, T])