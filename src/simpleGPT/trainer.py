import torch
from torch.nn import functional as F
import torch.distributed as dist
from simpleGPT.hellaswag import render_example, iterate_examples
from tqdm import tqdm
import math

def get_lr(it, max_lr, min_lr, warmup_steps, max_steps):
    if it < warmup_steps:
        return (it+1)/warmup_steps * max_lr
    
    if it > max_steps:
        return min_lr
    
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def jest_train_step(config, ddp, device, device_type, model, optimizer, step, dataloader, grad_accum_steps):
    model.train()
    optimizer.zero_grad()
    loss_accum = .0
    for accum_step in range(grad_accum_steps):
        x,y,ref_scores = dataloader.new_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (accum_step == (grad_accum_steps-1))
        if config.autocast_bf16 :
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        ## Scale the loss to compensate the gradient accumulation steps because the the loss is averaged
        # only across the bini-batch but not across the full accumulated batch
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    
    ### GRAD CLIP
    if config.use_grad_clip :
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    else:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        
    ### OPTIM STEP
    min_lr = config.max_lr * config.min_lr_ratio
    lr = get_lr(step, config.max_lr, min_lr, config.warmup_steps , config.max_steps)
    for param_group in optimizer.param_groups:
        param_group['lr']  = lr
    optimizer.step()

    torch.cuda.synchronize()
    if ddp: dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    
    return loss_accum,norm,lr


def validation_step(config, device, device_type, val_loader, model):
    model.eval()
    val_loader.reset()
    with torch.no_grad():
        val_loss_accum = .0
        for _ in range(config.val_n_steps):
            x, y = val_loader.new_batch()
            x, y = x.to(device), y.to(device)
            if config.autocast_bf16 :
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)
            loss = loss / config.val_n_steps
            val_loss_accum += loss.detach()
    return val_loss_accum

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

def hellaswag_eval_step(config, ddp, ddp_rank, ddp_world_size, device, device_type, model):
    num_correct_norm = 0
    num_total = 0
    n = 10_042
    for i, example in tqdm(enumerate(iterate_examples("val")), total=10_042):
        if i > n: break
            # only process examples where i % ddp_world_size == ddp_rank
        if i % ddp_world_size != ddp_rank:
            continue
            # render the example into tokens and labels
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
            # get the logits
        with torch.no_grad():
            if config.autocast_bf16 :
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
            else:
                logits, loss = model(tokens)
            pred_norm = get_most_likely_row(tokens, mask, logits)
        num_total += 1
        num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
    if ddp:
        num_total = torch.tensor(num_total, dtype=torch.long, device=device)
        num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
        torch.cuda.synchronize()
        dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        num_total = num_total.item()
        num_correct_norm = num_correct_norm.item()
    acc_norm = num_correct_norm / num_total
    return num_correct_norm,num_total,acc_norm