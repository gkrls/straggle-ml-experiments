#!/usr/bin/env python3
"""
Fixed GPT-2 training script for OpenWebText with proper hyperparameters
and training stability fixes. Optimized for 6 nodes with 1 P100 GPU each.
"""

import argparse
import os
import sys
import json
import time
import datetime
import random
import math
from pathlib import Path
import numpy as np
import warnings
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

# ------------------------- Parquet layout helpers ---------------------
def _resolve_parquet_layout(data_root: Path):
    r = data_root
    if (r / "parquet" / "train").is_dir() and (r / "parquet" / "val").is_dir():
        return str(r / "parquet" / "train" / "*.parquet"), str(r / "parquet" / "val" / "*.parquet"), False
    if (r / "train").is_dir() and (r / "val").is_dir():
        return str(r / "train" / "*.parquet"), str(r / "val" / "*.parquet"), False
    if list((r / "parquet").glob("*.parquet")):
        return str(r / "parquet" / "*.parquet"), None, True
    if list(r.glob("*.parquet")):
        return str(r / "*.parquet"), None, True
    raise FileNotFoundError(f"No Parquet files found under {r}.")

def hf_load_train_val_parquet(data_root: Path, val_fraction=0.0005, seed=42, cache_dir=None):
    train_glob, val_glob, split_needed = _resolve_parquet_layout(data_root)
    if split_needed:
        ds = load_dataset("parquet", data_files={"train": train_glob}, cache_dir=cache_dir)
        splits = ds["train"].train_test_split(test_size=val_fraction, seed=seed, shuffle=True)
        return splits["train"], splits["test"]
    else:
        ds = load_dataset("parquet", data_files={"train": train_glob, "validation": val_glob}, cache_dir=cache_dir)
        return ds["train"], ds["validation"]

# ------------------------- IterableDataset (DDP-aware) ----------------
class GPT2Windows(IterableDataset):
    """Streams (x,y) windows from docs with proper tokenization."""
    def __init__(self, hf_split, tokenizer, seq_len=1024, stride=1024,
                 world_size=1, rank=0, seed=42, append_eos=True):
        super().__init__()
        self.ds = hf_split
        self.tok = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.append_eos = append_eos
        self.PAD_ID = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.EOS_ID = self.tok.eos_token_id

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _encode_line(self, line: str):
        # Temporarily suppress the tokenizer warning about sequence length
        # We handle chunking ourselves in _yield_windows
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Token indices sequence length is longer than.*")
            # Also suppress the logging warning from transformers
            logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
            
            ids = self.tok.encode(line, add_special_tokens=False)
            
            # Reset logging level
            logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.WARNING)
        
        if self.append_eos and (not ids or ids[-1] != self.EOS_ID):
            ids.append(self.EOS_ID)
        return ids

    def _yield_windows(self, toks):
        # Window size is seq_len (inputs) + 1 (for targets)
        window_size = self.seq_len + 1
        for i in range(0, max(0, len(toks) - window_size + 1), self.stride):
            w = toks[i:i+window_size]
            if len(w) == window_size:
                # Return the full window - we'll split into inputs/targets later
                x = torch.tensor(w, dtype=torch.long)
                yield x

    def _index_permutation(self, n, rng: random.Random):
        if n <= 1:
            for j in range(n): yield j
            return
        start = rng.randrange(n)
        while True:
            step = rng.randrange(1, n)
            if math.gcd(step, n) == 1:
                break
        for j in range(n):
            yield (start + j * step) % n

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        num_workers = info.num_workers if info else 1
        worker_id = info.id if info else 0

        consumers = max(1, self.world_size * num_workers)
        consumer_id = self.rank * num_workers + worker_id

        rng = random.Random(1337 + self.seed + worker_id + self.epoch)
        n = len(self.ds)
        for i in self._index_permutation(n, rng):
            if (i % consumers) != consumer_id:
                continue
            ex = self.ds[i]
            toks = self._encode_line(ex["text"])
            yield from self._yield_windows(toks)

# ------------------------- AverageMeter ---------------------------
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = 0.0; self.count = 0.0; self.avg = 0.0
    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1.0, self.count)
    def all_reduce(self):
        if dist.is_available() and dist.is_initialized():
            backend = dist.get_backend()
            device = torch.device(f"cuda:{torch.cuda.current_device()}") if backend == dist.Backend.NCCL else torch.device("cpu")
            t = torch.tensor([self.sum, self.count], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            self.sum, self.count = t.cpu().tolist()
            self.avg = self.sum / max(1.0, self.count)

# ------------------------- Learning Rate Schedule ------------------
def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    """Cosine learning rate schedule with warmup"""
    # Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters  # +1 to avoid 0 lr at step 0
    # After lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# ------------------------- Validate --------------------------------
@torch.no_grad()
def validate(model, loader, device, args, max_batches=50):
    model.eval()
    losses = AverageMeter()
    
    for batch_idx, inputs in enumerate(loader):
        if batch_idx >= max_batches:
            break
        
        # inputs is seq_len+1, split properly    
        if inputs.size(1) != args.seq_len + 1:
            continue
            
        inputs_batch = inputs[:, :-1].contiguous().to(device, non_blocking=True)
        targets = inputs[:, 1:].contiguous().to(device, non_blocking=True)
        
        with torch.amp.autocast(device_type='cuda', enabled=args.amp):
            outputs = model(inputs_batch)
            logits = outputs.logits
            # Reshape for loss calculation
            B, T, V = logits.shape
            logits = logits.view(B*T, V)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        losses.update(loss.item(), B*T)
    
    losses.all_reduce()
    val_loss = losses.avg
    val_ppl = float(np.exp(np.clip(val_loss, 0, 20)))
    return {'val_loss': val_loss, 'val_ppl': val_ppl}

def now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ------------------------- Train 1 epoch --------------------------
def train_one_epoch(model, dataloader, optimizer, device, scaler, args, 
                    epoch, total_steps, warmup_iters, lr_decay_iters):
    model.train()
    
    losses = AverageMeter()
    step_time = AverageMeter()
    data_time = AverageMeter()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    epoch_start = time.perf_counter()
    
    step_start = time.perf_counter()
    tokens_seen = 0
    batches = 0
    
    for batch_idx, inputs in enumerate(dataloader):
        data_time.update(time.perf_counter() - step_start, n=1)
        
        # inputs is now seq_len+1, split into input and target
        if inputs.size(1) != args.seq_len + 1:
            print(f"[Warning] Unexpected sequence length: {inputs.size(1)}, expected {args.seq_len + 1}")
            continue
            
        # Calculate current step
        current_step = epoch * args.steps_per_epoch + batch_idx if args.steps_per_epoch else total_steps
        
        # Update learning rate
        lr = get_lr(current_step, warmup_iters, args.learning_rate, lr_decay_iters, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Split into inputs and targets
        inputs_batch = inputs[:, :-1].contiguous().to(device, non_blocking=True)
        targets = inputs[:, 1:].contiguous().to(device, non_blocking=True)
        
        # Forward pass
        with torch.amp.autocast(device_type='cuda', enabled=args.amp):
            outputs = model(inputs_batch)
            logits = outputs.logits
            # Reshape for loss calculation
            B, T, V = logits.shape
            logits = logits.view(B*T, V)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"[Warning] NaN loss detected at step {current_step}, skipping batch")
            continue
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            # Gradient clipping is CRITICAL
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        # Update metrics
        losses.update(loss.item(), B*T)
        tokens_seen += B*T
        batches += 1
        total_steps += 1
        
        step_time.update(time.perf_counter() - step_start, n=1)
        step_start = time.perf_counter()
        
        # Log progress
        if args.rank == 0:
            # During warmup, print every 50 steps
            if warmup_iters > 0 and current_step <= warmup_iters and batch_idx % 50 == 0:
                print(f"[{now()}][Epoch {epoch:03d}][Step {batch_idx:04d}/{args.steps_per_epoch:04d}] "
                      f"Loss: {loss.item():.4f} | LR: {lr:.6f} (warmup) | Grad Norm: {grad_norm:.2f}")
        
        if args.steps_per_epoch and batches >= args.steps_per_epoch:
            break
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    duration = time.perf_counter() - epoch_start
    
    tok_per_sec = tokens_seen / max(1e-6, duration)
    
    local_loss = losses.avg
    losses.all_reduce()
    train_loss_global = losses.avg
    train_ppl = float(np.exp(np.clip(train_loss_global, 0, 20)))
    
    return {
        'train_loss_global': train_loss_global,
        'train_loss': local_loss,
        'train_step_time': step_time.avg,
        'train_data_time': data_time.avg,
        'train_comp_time': step_time.avg - data_time.avg,
        'epoch_duration': duration,
        'epoch_throughput': tok_per_sec,
        'tokens_seen': tokens_seen,
        'train_ppl': train_ppl,
        'batches': batches,
        'total_steps': total_steps,
    }

def save_log(path, log):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)



# ------------------------- Train driver --------------------------
def train(args):
    device = torch.device(args.device)
    
    data_root = Path(args.data).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"--data path not found: {data_root}")
    
    cache_dir = Path(args.cache_dir) if args.cache_dir is not None else (data_root / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    
    # Tokenizer: GPT-2 BPE
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 uses eos_token as pad_token
    vocab_size = len(tokenizer)
    
    if args.rank == 0:
        print(f"Loading Parquet dataset from {data_root} (cache: {cache_dir}) ...", flush=True)
    
    # Use smaller validation fraction for faster validation
    ds_train, ds_val = hf_load_train_val_parquet(
        data_root, 
        val_fraction=args.val_fraction, 
        seed=42, 
        cache_dir=str(cache_dir)
    )
    
    # Create datasets with proper sequence length
    train_ds = GPT2Windows(
        ds_train, tokenizer, 
        seq_len=args.seq_len, 
        stride=args.seq_len,  # Non-overlapping for efficiency
        world_size=args.world_size, 
        rank=args.rank, 
        seed=args.seed
    )
    val_ds = GPT2Windows(
        ds_val, tokenizer, 
        seq_len=args.seq_len, 
        stride=args.seq_len,
        world_size=args.world_size, 
        rank=args.rank, 
        seed=args.seed + 1
    )
    
    def _seed_worker(worker_id):
        worker_seed = (args.seed + args.rank * max(1, args.workers) + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        persistent_workers=(args.workers > 0),
        worker_init_fn=_seed_worker
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        num_workers=min(2, args.workers),  # Fewer workers for validation
        pin_memory=True,
        worker_init_fn=_seed_worker
    )
    
    # Model configuration - GPT2-small (124M parameters)
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=args.seq_len,
        n_ctx=args.seq_len,
        n_embd=768,      # Standard GPT2-small dimensions
        n_layer=12,      # 12 layers for GPT2-small
        n_head=12,       # 12 heads
        n_inner=3072,    # 4x n_embd
        activation_function="gelu",
        resid_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,  # Important for stability
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Initialize model with proper weight initialization
    model = GPT2LMHeadModel(cfg)
    
    # Initialize weights properly (critical for stability)
    def init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    model.apply(init_weights)
    model.to(device)
    
    # Wrap in DDP
    model = DDP(
        model, 
        device_ids=[args.local_rank] if device.type == "cuda" else None,
        gradient_as_bucket_view=True, 
        find_unused_parameters=False,
        static_graph=args.static_graph
    )
    
    if args.rank == 0:
        n_tr = len(ds_train)
        n_va = len(ds_val)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nDATA ROOT: {data_root}")
        print(f"  Train docs: {n_tr:,} | Val docs: {n_va:,}")
        print(f"  Vocab size: {vocab_size:,}")
        print(f"  Seq len: {args.seq_len}")
        print(f"\nModel: GPT-2 small (124M parameters)")
        print(f"  Total parameters: {n_params:,}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Total batch size: {args.batch_size * args.world_size}")
        print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {args.batch_size * args.world_size * args.gradient_accumulation_steps}")
        print("=" * 60, flush=True)
    
    # Optimizer with proper betas for GPT training
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        betas=(0.9, 0.95),  # GPT-3 settings
        eps=1e-8,
        weight_decay=args.weight_decay
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp) if device.type == "cuda" and args.amp else None
    
    best_ppl = float('inf')
    total_steps = 0
    
    # Setup learning rate schedule parameters
    warmup_iters = args.warmup_iters
    lr_decay_iters = args.lr_decay_iters
    
    if args.rank == 0:
        if args.json is None:
            args.json = "gpt2_openwebtext_fixed.json"
        log = {
            "time": now(), 
            "data_root": str(data_root), 
            "cache_dir": str(cache_dir),
            "config": vars(args), 
            "vocab_size": vocab_size, 
            "epochs": {}
        }
        save_log(args.json, log)
    
    # Training loop
    for epoch in range(args.epochs):
        if args.rank == 0:
            if epoch == 0 and warmup_iters > 0:
                print(f"[{now()}][Epoch {epoch:03d}] Starting... first {warmup_iters} steps are warmup...", flush=True)
            else:
                print(f"[{now()}][Epoch {epoch:03d}] Starting...", flush=True)


        train_ds.set_epoch(epoch)  # Shuffle training data
        
        # Train for one epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, scaler, args,
            epoch, total_steps, warmup_iters, lr_decay_iters
        )
        total_steps = train_metrics['total_steps']
        
        # Validate
        val_metrics = validate(model, val_loader, device, args)
        
        epoch_time = train_metrics['epoch_duration']
        current_lr = optimizer.param_groups[0]['lr']
        
        if args.rank == 0:
            print(f"[{now()}][Epoch {epoch:03d}] "
                  f"train_loss={train_metrics['train_loss']:.4f} "
                  f"train_ppl={train_metrics['train_ppl']:.2f} "
                  f"val_loss={val_metrics['val_loss']:.4f} "
                  f"val_ppl={val_metrics['val_ppl']:.2f} "
                  f"lr={current_lr:.6f} time={epoch_time:.2f}s "
                  f"tok/s={train_metrics['epoch_throughput']:.0f}",
                  flush=True)
            
            # Log metrics
            epoch_metrics = {
                "train_loss": float(train_metrics['train_loss']),
                "train_loss_global": float(train_metrics['train_loss_global']),
                "train_step_time": float(train_metrics['train_step_time']),
                "train_data_time": float(train_metrics['train_data_time']),
                "train_comp_time": float(train_metrics['train_comp_time']),
                "train_duration": float(train_metrics['epoch_duration']),
                "train_throughput": float(train_metrics['epoch_throughput']),
                "val_loss": float(val_metrics['val_loss']),
                "val_ppl": float(val_metrics['val_ppl']),
                "lr": float(current_lr),
                "epoch_time": float(epoch_time),
                "epoch_throughput": float(train_metrics['epoch_throughput']),
                "steps": int(train_metrics['batches']),
            }
            
            with open(args.json, "r") as f:
                log = json.load(f)
            log["epochs"][str(epoch)] = epoch_metrics
            save_log(args.json, log)
            
            # Track best model
            if val_metrics['val_ppl'] < best_ppl:
                best_ppl = val_metrics['val_ppl']
                print(f"[{now()}] New best val_ppl: {best_ppl:.2f}")
    
    if args.rank == 0:
        print(f"\n[{now()}] Training completed!")
        print(f"Best validation perplexity: {best_ppl:.2f}")

# ------------------------- DDP setup / teardown -------------------
def setup_ddp(args):
    def env_int(k, d): return d if os.environ.get(k) in (None, "") else int(os.environ.get(k))
    def env_str(k, d): return d if os.environ.get(k) in (None, "") else os.environ.get(k)
    
    args.rank = env_int("RANK", args.rank)
    args.world_size = env_int("WORLD_SIZE", args.world_size)
    args.master_addr = env_str("MASTER_ADDR", args.master_addr)
    args.master_port = env_int("MASTER_PORT", args.master_port)
    args.iface = env_str("IFACE", args.iface)
    args.local_rank = (args.rank % torch.cuda.device_count()) if torch.cuda.device_count() else 0
    
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    
    os.environ.setdefault("RANK", str(args.rank))
    os.environ.setdefault("WORLD_SIZE", str(args.world_size))
    os.environ.setdefault("MASTER_ADDR", args.master_addr)
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", args.iface)
    os.environ.setdefault("GLOO_NSOCKS_PERTHREAD", "2")
    os.environ.setdefault("GLOO_BUFFSIZE", "8388608")
    
    # Use backend from args
    dist.init_process_group(
        backend=args.backend, 
        init_method="env://", 
        rank=args.rank,
        world_size=args.world_size, 
        timeout=datetime.timedelta(seconds=300)
    )
    
    if args.rank == 0:
        print(f"[DDP] backend={args.backend} world_size={args.world_size} "
              f"master={args.master_addr}:{args.master_port} iface={args.iface} local_rank={args.local_rank}", flush=True)

def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

# ------------------------- Entry ----------------------------------
def main():
    p = argparse.ArgumentParser("Fixed GPT-2 on OpenWebText (Parquet)")
    
    # DDP/system
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world_size', type=int, default=6, help='Number of GPUs (6 for your setup)')
    p.add_argument('--iface', type=str, default="ens4f0", help='Network interface')
    p.add_argument('--master_addr', type=str, default="127.0.0.1")
    p.add_argument("--master_port", type=int, default=29500)
    p.add_argument("--backend", type=str, default="nccl", choices=['gloo', 'nccl'], help='DDP backend')
    p.add_argument("--device", type=str, choices=['cuda','cpu'], default='cuda')
    p.add_argument("--deterministic", action='store_true')
    p.add_argument("--static_graph", action='store_true', help='DDP static graph optimization')
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--json", type=str, default=None)
    p.add_argument('--seed', type=int, default=1337)
    
    # Data
    p.add_argument('--data', type=str, required=True, help='Path to OpenWebText parquet files')
    p.add_argument('--cache_dir', type=str, default=None)
    p.add_argument('--val_fraction', type=float, default=0.0005, help='Fraction for validation')
    p.add_argument('--tokenizer', type=str, default="gpt2", help='Tokenizer to use')
    
    # Training hyperparameters (FIXED for stability)
    p.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    p.add_argument('--steps_per_epoch', type=int, default=500, help='Steps per epoch (500 = ~15 min)')
    p.add_argument('--batch_size', type=int, default=12, help='Batch size per GPU')
    p.add_argument('--gradient_accumulation_steps', type=int, default=5, help='Gradient accumulation')
    p.add_argument('--learning_rate', type=float, default=6e-4, help='Max learning rate')
    p.add_argument('--min_lr', type=float, default=6e-5, help='Min learning rate')
    p.add_argument('--warmup_iters', type=int, default=200, help='Warmup iterations')
    p.add_argument('--lr_decay_iters', type=int, default=5000, help='LR decay iterations')
    p.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    p.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    p.add_argument("--amp", action="store_true", default=True, help='Use mixed precision')
    p.add_argument("--prefetch_factor", type=int, default=2)
    
    # Model configuration
    p.add_argument('--seq_len', type=int, default=1024, help='Sequence length')
    p.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Parse arguments
    args = p.parse_args()
    
    # Set deterministic behavior if requested
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    else:
        torch.backends.cudnn.benchmark = True
    
    # CUDA optimizations
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Fallback to CPU if CUDA not available
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        print("[Info] Using device=cpu because CUDA is not available", flush=True)
    
    if args.amp and args.device == 'cpu':
        args.amp = False
        print("[Info] Disabling AMP because CUDA is not available", flush=True)
    
    if args.workers < 0:
        args.workers = 0
    
    # Setup stdout
    sys.stdout.reconfigure(line_buffering=True)
    
    # Setup DDP
    setup_ddp(args)
    
    if args.rank == 0:
        print("Configuration:")
        print(json.dumps(vars(args), indent=2))
    
    try:
        train(args)
    finally:
        cleanup()

if __name__ == "__main__":
    main()