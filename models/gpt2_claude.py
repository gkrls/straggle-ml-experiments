#!/usr/bin/env python3
"""
Distributed GPT-2 Training on OpenWebText using HuggingFace
Designed for 6 nodes with 1 GPU (P100) each
"""

import os
import sys
import argparse
import time
import datetime
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW

from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from datasets import load_dataset


# ------------------------- AverageMeter for metrics -------------------------
class AverageMeter:
    """Computes and stores average and current value with DDP support"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sum = 0.0
        self.count = 0.0
        self.avg = 0.0
    
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


# ------------------------- Learning Rate Schedule ----------------------------
def get_lr(update_idx, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    """Cosine learning rate schedule with linear warmup"""
    # Linear warmup
    if update_idx < warmup_iters:
        return learning_rate * (update_idx + 1) / max(1, warmup_iters)
    # After decay iters, return min learning rate
    if update_idx > lr_decay_iters:
        return min_lr
    # Cosine decay
    decay_ratio = (update_idx - warmup_iters) / max(1, (lr_decay_iters - warmup_iters))
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ------------------------- OpenWebText Dataset -------------------------------
class OpenWebTextDataset(IterableDataset):
    """Streaming dataset for OpenWebText with DDP support"""
    def __init__(self, split, tokenizer, seq_len=1024, world_size=1, rank=0, seed=42):
        self.dataset = load_dataset('openwebtext', split=split, streaming=True)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.epoch = 0
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    def __iter__(self):
        # Setup worker info for multi-worker data loading
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        
        # Total consumers across all ranks and workers
        total_consumers = self.world_size * num_workers
        consumer_id = self.rank * num_workers + worker_id
        
        # Buffer for accumulating tokens
        buffer = []
        
        # Iterate through dataset, each consumer takes its share
        for i, example in enumerate(self.dataset):
            # Distribute examples across consumers
            if i % total_consumers != consumer_id:
                continue
            
            # Tokenize text
            tokens = self.tokenizer.encode(
                example['text'], 
                add_special_tokens=False,
                truncation=False
            )
            
            # Add EOS token if not present
            if not tokens or tokens[-1] != self.tokenizer.eos_token_id:
                tokens.append(self.tokenizer.eos_token_id)
            
            buffer.extend(tokens)
            
            # Yield complete sequences
            while len(buffer) >= self.seq_len + 1:
                yield torch.tensor(buffer[:self.seq_len + 1], dtype=torch.long)
                buffer = buffer[self.seq_len:]  # Non-overlapping for training


# ------------------------- Training Functions --------------------------------
@torch.no_grad()
def validate(model, loader, device, args, max_batches=200):
    """Validation loop"""
    model.eval()
    losses = AverageMeter()
    
    for batch_idx, inputs in enumerate(loader):
        if batch_idx >= max_batches:
            break
        
        x = inputs[:, :-1].to(device, non_blocking=True)
        y = inputs[:, 1:].to(device, non_blocking=True)
        
        with torch.amp.autocast(device_type='cuda', enabled=args.amp):
            outputs = model(x, labels=y)
            loss = outputs.loss
        
        B, T = y.shape
        losses.update(loss.item(), B * T)
    
    losses.all_reduce()
    val_loss = losses.avg
    val_ppl = float(np.exp(np.clip(val_loss, 0, 20)))
    
    return {'val_loss': val_loss, 'val_ppl': val_ppl}


def train_one_epoch(model, dataloader, optimizer, device, scaler, args,
                    epoch, total_updates, warmup_iters, lr_decay_iters):
    """Train for one epoch with gradient accumulation"""
    model.train()
    
    losses = AverageMeter()
    step_time = AverageMeter()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    epoch_start = time.perf_counter()
    
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    num_updates = total_updates
    tokens_seen = 0
    batches = 0
    
    step_start = time.perf_counter()
    
    for batch_idx, inputs in enumerate(dataloader):
        # Prepare batch
        x = inputs[:, :-1].to(device, non_blocking=True)
        y = inputs[:, 1:].to(device, non_blocking=True)
        
        # Update learning rate based on completed optimizer updates
        lr = get_lr(num_updates, warmup_iters, args.learning_rate, lr_decay_iters, args.min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        # Determine if we should sync gradients
        ddp_sync = (micro_step == args.gradient_accumulation_steps - 1)
        sync_context = nullcontext() if ddp_sync else model.no_sync()
        
        # Forward and backward pass
        with sync_context:
            with torch.amp.autocast(device_type='cuda', enabled=args.amp):
                outputs = model(x, labels=y)
                loss_full = outputs.loss
                loss = loss_full / args.gradient_accumulation_steps
                
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        
        B, T = x.shape
        losses.update(loss_full.item(), B * T)
        tokens_seen += B * T
        micro_step += 1
        batches += 1
        
        # Optimizer step after accumulation
        if ddp_sync:
            if scaler is not None:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            num_updates += 1
            micro_step = 0
            
            # Logging during warmup
            if args.rank == 0 and warmup_iters > 0 and num_updates <= warmup_iters:
                if num_updates % 50 == 0:
                    print(f"[Warmup][Update {num_updates}/{warmup_iters}] "
                          f"Loss: {loss_full.item():.4f} | LR: {lr:.6f} | "
                          f"Grad Norm: {grad_norm:.2f}")
        
        step_time.update(time.perf_counter() - step_start)
        step_start = time.perf_counter()
        
        # Optional: limit steps per epoch
        if args.steps_per_epoch and batches >= args.steps_per_epoch:
            break
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    duration = time.perf_counter() - epoch_start
    
    # Calculate metrics
    losses.all_reduce()
    train_loss = losses.avg
    train_ppl = float(np.exp(np.clip(train_loss, 0, 20)))
    throughput = tokens_seen / max(1e-6, duration)
    
    return {
        'train_loss': train_loss,
        'train_ppl': train_ppl,
        'throughput': throughput,
        'duration': duration,
        'total_updates': num_updates,
        'tokens_seen': tokens_seen,
        'lr': lr
    }


# ------------------------- DDP Setup/Teardown --------------------------------
def setup_ddp(args):
    """Initialize distributed training with comprehensive configuration"""
    # Get environment variables with defaults
    def env_int(k, d):
        return d if os.environ.get(k) in (None, "") else int(os.environ.get(k))
    def env_str(k, d):
        return d if os.environ.get(k) in (None, "") else os.environ.get(k)
    
    args.rank = env_int("RANK", args.rank)
    args.world_size = env_int("WORLD_SIZE", args.world_size)
    args.master_addr = env_str("MASTER_ADDR", args.master_addr)
    args.master_port = env_int("MASTER_PORT", args.master_port)
    args.local_rank = env_int("LOCAL_RANK", args.rank % torch.cuda.device_count() if torch.cuda.device_count() else 0)
    
    # Set CUDA device
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    
    # Set environment variables for DDP
    os.environ.setdefault("RANK", str(args.rank))
    os.environ.setdefault("WORLD_SIZE", str(args.world_size))
    os.environ.setdefault("MASTER_ADDR", args.master_addr)
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    
    # Network interface configuration
    if args.iface:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", args.iface)
        os.environ.setdefault("NCCL_SOCKET_IFNAME", args.iface)
    
    # Additional optimizations
    os.environ.setdefault("GLOO_NSOCKS_PERTHREAD", "2")
    os.environ.setdefault("GLOO_BUFFSIZE", "8388608")
    
    # Initialize process group
    dist.init_process_group(
        backend=args.backend,
        init_method="env://",
        rank=args.rank,
        world_size=args.world_size,
        timeout=datetime.timedelta(seconds=300)
    )
    
    if args.rank == 0:
        print(f"[DDP Setup] backend={args.backend}, world_size={args.world_size}, "
              f"master={args.master_addr}:{args.master_port}, iface={args.iface}")


def cleanup():
    """Cleanup distributed training"""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ------------------------- Main Training Function ----------------------------
def train(args):
    """Main training loop"""
    device = torch.device(args.device)
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    
    # Create datasets
    if args.rank == 0:
        print("Loading OpenWebText dataset...")
    
    train_dataset = OpenWebTextDataset(
        'train', tokenizer, args.seq_len, 
        args.world_size, args.rank, args.seed
    )
    
    # For validation, we'll use a small subset
    val_dataset = OpenWebTextDataset(
        'train', tokenizer, args.seq_len,
        args.world_size, args.rank, args.seed + 1000
    )
    
    # Data loaders
    def seed_worker(worker_id):
        worker_seed = (args.seed + args.rank * args.workers + worker_id) % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=(args.workers > 0)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=min(2, args.workers),
        pin_memory=True,
        worker_init_fn=seed_worker
    )
    
    # Model configuration - matching your GPT-2 small setup
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=args.seq_len,
        n_ctx=args.seq_len,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=3072,
        activation_function="gelu",
        resid_pdrop=args.dropout,
        attn_pdrop=0.1,
        embd_pdrop=args.dropout,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Initialize model
    model = GPT2LMHeadModel(config)
    
    # Custom weight initialization (matching your script)
    def init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
    
    model.apply(init_weights)
    model.to(device)
    
    # Wrap model in DDP
    model = DDP(
        model,
        device_ids=[args.local_rank] if device.type == "cuda" else None,
        gradient_as_bucket_view=True,
        find_unused_parameters=False,
        static_graph=args.static_graph
    )
    
    if args.rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel: GPT-2 small (124M parameters)")
        print(f"  Total parameters: {n_params:,}")
        print(f"  Sequence length: {args.seq_len}")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"  Effective batch size: {args.batch_size * args.world_size * args.gradient_accumulation_steps}")
        print("=" * 60)
    
    # Optimizer with parameter groups (matching your weight decay strategy)
    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:  # Weight matrices
            decay_params.append(param)
        else:  # Biases and LayerNorm parameters
            nodecay_params.append(param)
    
    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Mixed precision scaler
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp) if device.type == "cuda" and args.amp else None
    
    # Training schedule
    updates_per_epoch = max(1, args.steps_per_epoch // args.gradient_accumulation_steps) if args.steps_per_epoch else 1000
    total_updates_planned = args.epochs * updates_per_epoch
    warmup_iters = args.warmup_iters
    lr_decay_iters = args.lr_decay_iters if args.lr_decay_iters > 0 else total_updates_planned
    
    if args.rank == 0:
        print(f"Training schedule: warmup {warmup_iters} updates, decay to {lr_decay_iters} updates")
    
    # Training loop
    best_ppl = float('inf')
    total_updates = 0
    
    for epoch in range(args.epochs):
        if args.rank == 0:
            print(f"\n[Epoch {epoch+1}/{args.epochs}] Starting...")
        
        train_dataset.set_epoch(epoch)
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, scaler, args,
            epoch, total_updates, warmup_iters, lr_decay_iters
        )
        total_updates = train_metrics['total_updates']
        
        # Validate
        val_metrics = validate(model, val_loader, device, args, max_batches=200)
        
        # Log results
        if args.rank == 0:
            print(f"[Epoch {epoch+1}] "
                  f"train_loss={train_metrics['train_loss']:.4f} "
                  f"train_ppl={train_metrics['train_ppl']:.2f} "
                  f"val_loss={val_metrics['val_loss']:.4f} "
                  f"val_ppl={val_metrics['val_ppl']:.2f} "
                  f"lr={train_metrics['lr']:.6f} "
                  f"throughput={train_metrics['throughput']:.0f} tok/s "
                  f"time={train_metrics['duration']:.1f}s")
            
            if val_metrics['val_ppl'] < best_ppl:
                best_ppl = val_metrics['val_ppl']
                print(f"  New best validation perplexity: {best_ppl:.2f}")
    
    if args.rank == 0:
        print(f"\nTraining completed! Best validation perplexity: {best_ppl:.2f}")


# ------------------------- Main Entry Point ----------------------------------
def main():
    parser = argparse.ArgumentParser(description='Distributed GPT-2 Training on OpenWebText')
    
    # DDP/System arguments (matching your configuration)
    parser.add_argument('--rank', type=int, default=0, help='Rank of current process')
    parser.add_argument('--world_size', type=int, default=6, help='Number of processes/GPUs')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1', help='Master node address')
    parser.add_argument('--master_port', type=int, default=29500, help='Master port')
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'])
    parser.add_argument('--iface', type=str, default='ens4f0', help='Network interface')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--static_graph', action='store_true', help='Enable DDP static graph optimization')
    parser.add_argument('--workers', type=int, default=4, help='Data loader workers')
    parser.add_argument('--seed', type=int, default=1337, help='Random seed')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic mode')
    
    # Data arguments
    parser.add_argument('--tokenizer', type=str, default='gpt2', help='Tokenizer name')
    parser.add_argument('--json', type=str, default=None, help='JSON log file path')
    
    # Training arguments (matching your hyperparameters)
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=500, help='Steps per epoch (micro-batches)')
    parser.add_argument('--batch_size', type=int, default=12, help='Batch size per GPU')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=5, help='Gradient accumulation')
    parser.add_argument('--learning_rate', type=float, default=6e-4, help='Maximum learning rate')
    parser.add_argument('--min_lr', type=float, default=6e-5, help='Minimum learning rate')
    parser.add_argument('--warmup_steps', type=int, default=200, help='Warmup updates')
    parser.add_argument('--lr_decay_iters', type=int, default=-1, help='LR decay updates (-1 for auto)')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    
    # Model arguments
    parser.add_argument('--seq_len', type=int, default=1024, help='Sequence length')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    args = parser.parse_args()
    
    # Set deterministic mode if requested
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
    
    # Fallback to CPU if needed
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        args.amp = False
        print("[Warning] CUDA not available, using CPU")
    
    # Line-buffered stdout
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except:
        pass
    
    # Setup DDP
    setup_ddp(args)
    
    if args.rank == 0:
        print(f"[{now()}] Configuration:")
        print(json.dumps(vars(args), indent=2), flush=True)
    
    try:
        train(args)
    finally:
        cleanup()


if __name__ == "__main__":
    main()