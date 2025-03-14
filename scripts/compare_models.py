import torch

from simpleGPT.simplegpt import GPT as a_GPT
from simpleGPT.simplegpt import GPTConfig as a_GPTConfig
from karpathy_gpt import GPT as k_GPT
from karpathy_gpt import GPTConfig as k_GPTConfig


def setup_seed(seed=1234567890):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def compare_optimizers(optimizer_a, optimizer_k):
    assert type(optimizer_a) == type(optimizer_k), f"Optimizer types do not match: {type(optimizer_a)} vs {type(optimizer_k)}"
    
    # Check the parameter groups
    print("Comparing parameter groups")
    assert len(optimizer_a.param_groups) == len(optimizer_k.param_groups), "Number of parameter groups do not match"

    for idx, (group1, group2) in enumerate(zip(optimizer_a.param_groups, optimizer_k.param_groups)):
        print(f"Comparing parameter group {idx}")
        assert group1.keys() == group2.keys(), f"Parameter group keys do not match: {group1.keys()} vs {group2.keys()}"
        for key in group1:
            if key == 'params':
                print(f"  Comparing 'params' key in group {idx}")
                assert len(group1[key]) == len(group2[key]), "Number of parameters in groups do not match"
                for p_idx, (p1, p2) in enumerate(zip(group1[key], group2[key])):
                    assert torch.equal(p1, p2), f"Parameter tensors do not match at index {p_idx}"
            else:
                print(f"  Comparing '{key}' key in group {idx}")
                assert group1[key] == group2[key], f"Group attribute {key} does not match: {group1[key]} vs {group2[key]}"

def compare_logits(logits1, logits2):
    # Compare outputs
    print("Comparing model outputs...")
    if torch.allclose(logits1, logits2, atol=1e-5):
        print("Test passed: Outputs are the same!")
    else:
        print("Test failed: Outputs differ!")
        differences = logits1 - logits2
        print("Output differences (logits1 - logits2):", differences)

def compare_parameters(model1, model2):
    all_match = True
    for name1, param1 in model1.named_parameters():
        param2 = dict(model2.named_parameters())[name1]
        if not torch.equal(param1, param2):
            print(f"Parameter mismatch: {name1} does not match.")
            all_match = False
    if all_match:
        print("All parameters match between the two models.")
    return all_match

def test_gpt_implementation():
    device_type = 'cuda'

    config = k_GPTConfig(block_size=64, n_layer=6, n_head=6)
    wd = 0.1
    lr = 6e-4

    print("####### Compare models at initialization...")
    # Initialize both models
    setup_seed()
    model_a = a_GPT(a_GPTConfig(block_size=64, n_layer=6, n_head=6))
    model_a.to('cuda')
    optimizer_a = model_a.configure_optimizers(weight_decay=wd, learning_rate=lr, device_type=device_type)
    setup_seed()
    model_k = k_GPT(config)
    model_k.to('cuda')
    optimizer_k = model_k.configure_optimizers(weight_decay=wd, learning_rate=lr, device_type=device_type)

    # Compare parameters
    print("Comparing model parameters...")
    if not compare_parameters(model_a, model_k):
        print("Test failed: Model parameters do not match!")
        return
    else:
        print("Model parameters match successfully.")

    # Generate random inputs
    batch_size = 1
    sequence_length = config.block_size  # Using maximum block size defined in the config
    random_input = torch.randint(config.vocab_size, (batch_size, sequence_length), dtype=torch.long).to('cuda')

    # Get outputs from both models
    model_a.eval()
    model_k.eval()
    logits1, _ = model_a(random_input)
    logits2, _ = model_k(random_input)

    # Compare parameters
    print("Comparing logits...")
    compare_logits(logits1, logits2)

    # Check if both optimizers are of the same type
    print("Comparing optimizer types")
    compare_optimizers(optimizer_a, optimizer_k)

    print("####### Testing after 10 optim step with random input")
    print('Step for model A')
    setup_seed()
    model_a.train()
    B, T = 1, config.block_size
    buf = torch.randint(0, config.vocab_size, size = (B*T+1,))
    x = (buf[:-1]).view(B, T).to('cuda')
    y = (buf[1:]).view(B, T).to('cuda')
    for _ in range(10):
        logits_a, loss_a = model_a(x, y)
        loss_a.backward()
        norm = torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
        for param_group in optimizer_a.param_groups:
            param_group['lr'] = lr
        optimizer_a.step()

    print('Step for model K')
    setup_seed()
    model_k.train()
    B, T = 1, config.block_size
    buf = torch.randint(0, config.vocab_size, size = (B*T+1,))
    x = (buf[:-1]).view(B, T).to('cuda')
    y = (buf[1:]).view(B, T).to('cuda')
    for _ in range(10):
        logits_k, loss_k = model_k(x, y)
        loss_k.backward()
        norm = torch.nn.utils.clip_grad_norm_(model_k.parameters(), 1.0)
        for param_group in optimizer_k.param_groups:
            param_group['lr'] = lr
        optimizer_k.step()

    # Compare parameters
    print("### Comparing logits...")
    compare_logits(logits_a, logits_k)

    # Check if both optimizers are of the same type
    print("### Comparing optimizer types")
    compare_optimizers(optimizer_a, optimizer_k)

        # Compare parameters
    print("### Comparing model parameters...")
    if not compare_parameters(model_a, model_k):
        print("Test failed: Model parameters do not match!")
        return
    else:
        print("Model parameters match successfully.")


test_gpt_implementation()
