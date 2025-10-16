"""
Bifrost training: bidirectional self-consistency pretraining.

Run as:
python bifrost_train.py --bifrost_enabled=True

Or distributed:
torchrun --nproc_per_node=8 bifrost_train.py -- --bifrost_enabled=True

Для отключения Bifrost (baseline):
torchrun --nproc_per_node=8 bifrost_train.py -- --bifrost_enabled=False
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import wandb
import torch

from nanochat.gpt import GPTBifrost, GPTBifrostConfig
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model
print_banner()

# -----------------------------------------------------------------------------
# User settings (такие же как в base_train.py + параметры Bifrost)
run = "dummy"
depth = 12
max_seq_len = 2048
num_iterations = -1
target_flops = -1.0
target_param_data_ratio = 20
device_batch_size = 32
total_batch_size = 524288
embedding_lr = 0.2
unembedding_lr = 0.004
weight_decay = 0.0
matrix_lr = 0.02
grad_clip = 1.0

# === BIFROST ПАРАМЕТРЫ ===
bifrost_enabled = True # Главный переключатель
bifrost_alpha = 0.2 # Вес consistency loss
bifrost_lambda = 0.2 # Дополнительный множитель
bifrost_beta = 0.6 # Баланс fwd/bwd (0.5 = равный вес)

eval_every = 250
eval_tokens = 20*524288
core_metric_every = 2000
core_metric_max_per_task = 500
sample_every = 2000
model_tag = "" # Если пусто, будет "d<depth>_bifrost" или "d<depth>_baseline"

config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# Compute init
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

# wandb logging
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project="nanochat-bifrost", 
    name=run, 
    config=user_config
)

# Tokenizer
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model config (добавляем Bifrost параметры)
num_layers = depth
model_dim = depth * 64
num_heads = max(1, (model_dim + 127) // 128)
num_kv_heads = num_heads

print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")
print0(f"Bifrost enabled: {bifrost_enabled}")
if bifrost_enabled:
    print0(f"Bifrost params: alpha={bifrost_alpha}, lambda={bifrost_lambda}, beta={bifrost_beta}")

model_config_kwargs = dict(
    sequence_len=max_seq_len,
    vocab_size=vocab_size,
    n_layer=num_layers,
    n_head=num_heads,
    n_kv_head=num_kv_heads,
    n_embd=model_dim,
    # Bifrost specific
    bifrost_enabled=bifrost_enabled,
    bifrost_alpha=bifrost_alpha,
    bifrost_lambda=bifrost_lambda,
    bifrost_beta=bifrost_beta,
)

# Batch size calculations
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# Initialize Model
with torch.device("meta"):
    model_config = GPTBifrostConfig(**model_config_kwargs)
    model = GPTBifrost(model_config)

model.to_empty(device="cuda")
model.init_weights()
orig_model = model
model = torch.compile(model, dynamic=False)

num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate num_iterations
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")

total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_tokens / num_params:.2f}")
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# Optimizer
optimizers = model.setup_optimizers(
    unembedding_lr=unembedding_lr,
    embedding_lr=embedding_lr,
    matrix_lr=matrix_lr,
    weight_decay=weight_decay
)
adamw_optimizer, muon_optimizer = optimizers

# DataLoaders
train_loader = tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="train")
build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val")
x, y = next(train_loader)

# LR scheduler
warmup_ratio = 0.0
warmdown_ratio = 0.2
final_lr_frac = 0.0

def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

def get_muon_momentum(it):
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95

# -----------------------------------------------------------------------------
# Training loop
min_val_bpb = float("inf")
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0

for step in range(num_iterations + 1):
    last_step = step == num_iterations
    flops_so_far = num_flops_per_token * total_batch_size * step
    
    # Validation
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        min_val_bpb = min(min_val_bpb, val_bpb)
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()
    
    # CORE metric evaluation
    if last_step or (step > 0 and step % core_metric_every == 0):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()
    
    # Sampling
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
        ]
        engine = Engine(orig_model, tokenizer)
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()
    
    # Save checkpoint
    if master_process and last_step:
        base_dir = get_base_dir()
        if not model_tag:
            suffix = "bifrost" if bifrost_enabled else "baseline"
            model_tag = f"d{depth}_{suffix}"
        checkpoint_dir = os.path.join(base_dir, "bifrost_checkpoints", model_tag)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            [opt.state_dict() for opt in optimizers],
            {
                "step": step,
                "val_bpb": val_bpb,
                "model_config": model_config_kwargs,
                "user_config": user_config,
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
            }
        )
        print0(f"✅ Saved checkpoint to {checkpoint_dir}")
    
    if last_step:
        break
    
    # === TRAINING STEP ===
    torch.cuda.synchronize()
    t0 = time.time()
    
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y = next(train_loader)
    
    # Gradient clipping
    if grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
    
    # Optimizer step
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    
    torch.cuda.synchronize()
    dt = time.time() - t0
    
    # Logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(world_tokens_per_fwdbwd / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100
    
    if step > 10:
        total_training_time += dt
    
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f}")
    
    # Log Bifrost metrics
    if step % 100 == 0:
        log_dict = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        
        # Добавляем Bifrost-специфичные метрики
        if bifrost_enabled and hasattr(orig_model, 'last_metrics') and orig_model.last_metrics:
            log_dict.update({
                "bifrost/loss_acc": orig_model.last_metrics['loss_acc'].item(),
                "bifrost/loss_cons": orig_model.last_metrics['loss_cons'].item(),
                "bifrost/loss_fwd": orig_model.last_metrics['loss_fwd'].item(),
                "bifrost/loss_bwd": orig_model.last_metrics['loss_bwd'].item(),
            })
        
        wandb_run.log(log_dict)

# Final stats
print0(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from nanochat.report import get_report
get_report().log(section=f"Bifrost training ({'enabled' if bifrost_enabled else 'baseline'})", data=[
    user_config,
    {
        "Number of parameters": num_params,
        "Number of iterations": num_iterations,
        "Minimum validation bpb": min_val_bpb,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results["core_metric"],
        "Total training time": f"{total_training_time/60:.2f}m",
    }
])

wandb_run.finish()
compute_cleanup()