import torch
from simpleGPT.simplegpt import GPT as a_GPT
from simpleGPT.simplegpt import GPTConfig as a_GPTConfig
from karpathy_gpt import GPT as k_GPT
from karpathy_gpt import GPTConfig as k_GPTConfig

def setup_seed(seed=1234567890):
    print(f"Setting random seed to {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def compare_optimizers():
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    master_process = True

    config = k_GPTConfig()
    wd = 0.1
    lr = 6e-4

    # Initialize both models with the same seed to ensure reproducibility
    print("Initializing model_a from simpleGPT")
    setup_seed()
    model_a = a_GPT(a_GPTConfig())
    model_a.to(device_type)
    optimizer1 = model_a.configure_optimizers(weight_decay=wd, learning_rate=lr, device_type=device_type)

    print("Initializing model_k from karpathy_gpt")
    setup_seed()
    model_k = k_GPT(k_GPTConfig())
    model_k.to(device_type)
    optimizer2 = model_k.configure_optimizers(weight_decay=wd, learning_rate=lr, device_type=device_type)

    # Check if both optimizers are of the same type
    print("Comparing optimizer types")
    assert type(optimizer1) == type(optimizer2), f"Optimizer types do not match: {type(optimizer1)} vs {type(optimizer2)}"
    
    # Check the parameter groups
    print("Comparing parameter groups")
    assert len(optimizer1.param_groups) == len(optimizer2.param_groups), "Number of parameter groups do not match"

    for idx, (group1, group2) in enumerate(zip(optimizer1.param_groups, optimizer2.param_groups)):
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

    print("All checks passed! The optimizers are equivalent.")

compare_optimizers()
