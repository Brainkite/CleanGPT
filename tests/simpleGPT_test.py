import pytest
import torch
from src.simpleGPT.simplegpt import GPTConfig
from src.simpleGPT.simplegpt import CasualSelfAttention, MLP, Block, GPT

class TestCasualSelfAttention:
    
    def test_initializes_without_errors_with_valid_config(self):
        config = GPTConfig()
        try:
            attention = CasualSelfAttention(config)
        except Exception as e:
            pytest.fail(f"Initialization failed with exception: {e}")
    
    def test_forward_outputs_proper_size_tensor(self):
        config = GPTConfig()
        attention = CasualSelfAttention(config)
        x = torch.randn(2, config.block_size, config.n_embd)
        
        out = attention(x)
        
        assert out.shape == (2, config.block_size, config.n_embd)


class TestMLP:
    
    def test_initializes_without_errors_with_valid_config(self):
        config = GPTConfig()
        try:
            mlp = MLP(config)
        except Exception as e:
            pytest.fail(f"Initialization failed with exception: {e}")
    
    def test_forward_outputs_proper_size_tensor(self):
        config = GPTConfig()
        mlp = MLP(config)
        x = torch.randn(2, config.block_size, config.n_embd)
        
        out = mlp(x)
        
        assert out.shape == (2, config.block_size, config.n_embd)


class TestBlock:
    
    def test_initializes_without_errors_with_valid_config(self):
        config = GPTConfig()
        try:
            block = Block(config)
        except Exception as e:
            pytest.fail(f"Initialization failed with exception: {e}")
            
    def test_forward_outputs_proper_size_tensor(self):
        config = GPTConfig()
        block = Block(config)
        x = torch.randn(2, config.block_size, config.n_embd)
        
        out = block(x)
        
        assert out.shape == (2, config.block_size, config.n_embd)
        
    
class TestGPT:
    
    def test_initializes_without_errors_with_valid_config(self):
        config = GPTConfig()
        try:
            gpt = GPT(config)
        except Exception as e:
            pytest.fail(f"Initialization failed with exception: {e}")
            
    def test_from_pretrained_passes_with_gpt2_model_type(self):
        try:
            gpt = GPT.from_pretrained('gpt2')
        except Exception as e:
            pytest.fail(f"Pretrained model crxeation failed: {e}")