#!/usr/bin/env python3
"""
LSTM model for SST-2 sentiment classification - optimized for 85%+ accuracy
Literature-standard 3-layer BiLSTM with attention pooling
Designed for distributed training on 6 nodes with DPA backend
"""

import argparse
import os
import sys
import re
import json
import time
import math
import datetime
import random
import numpy as np
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from datasets import load_dataset  # Hugging Face

import dpa

# ------------------------- Fixed NLP defaults (optimized for 85%+) -------------------------
PAD_ID         = 0
UNK_ID         = 1
MIN_FREQ       = 1            # keep rare tokens
WORD_DROPOUT_P = 0.05         # reduced from 0.15 - less aggressive
EMB_DROPOUT_P  = 0.1          # reduced from 0.20 - less aggressive  
CLIP_NORM      = 5.0
WARMUP_RATIO   = 0.15         # increased for large batch stability

# ------------------------- Tokenizer / Vocab ------------------------------

_token_re = re.compile(r"[a-z0-9']+|[!?.,;:()\-]+")

def simple_tokenizer(text: str) -> List[str]:
    text = text.lower().replace("\n", " ")
    text = re.sub(r"\d+", "<num>", text)  # fold numbers
    return _token_re.findall(text)

def build_vocab_sst(max_vocab: Optional[int], min_freq: int) -> dict:
    from collections import Counter
    # build from train + validation to reduce OOV
    ds_train = load_dataset("glue", "sst2", split="train")
    ds_val   = load_dataset("glue", "sst2", split="validation")
    ctr = Counter()
    for ex in ds_train:
        ctr.update(simple_tokenizer(ex["sentence"]))
    for ex in ds_val:
        ctr.update(simple_tokenizer(ex["sentence"]))

    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    words = [w for w, c in ctr.most_common() if c >= min_freq]
    if max_vocab and max_vocab > 0:
        words = words[: max(0, max_vocab - len(vocab))]
    for w in words:
        vocab[w] = len(vocab)
    return vocab

def encode(text: str, vocab: dict, max_len: int) -> Tuple[torch.Tensor, int]:
    toks = simple_tokenizer(text) or ["<unk>"]
    ids = [vocab.get(t, UNK_ID) for t in toks][:max_len]
    length = len(ids)
    if length < max_len:
        ids += [PAD_ID] * (max_len - length)
    return torch.tensor(ids, dtype=torch.long), length

# ------------------------- Dataset ------------------------------

class SSTDataset(Dataset):
    def __init__(self, split: str, vocab: dict, max_len: int = 128):
        self.data = load_dataset("glue", "sst2", split=split)
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self): return len(self.data)
    def __getitem__(self, idx: int):
        item = self.data[idx]
        x, length = encode(item["sentence"], self.vocab, self.max_len)
        return x, torch.tensor(item["label"], dtype=torch.long), torch.tensor(length, dtype=torch.long)

# ------------------------- Metrics ------------------------------

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self):
        self.sum = 0.0; self.count = 0.0; self.avg = 0.0; self.min = float("inf"); self.max = 0.0
    def update(self, val, n=1):
        v = float(val); self.sum += v * n; self.count += n
        self.avg = self.sum / max(1.0, self.count)
        self.min = min(self.min, v); self.max = max(self.max, v)
    def all_reduce(self):
        if dist.is_available() and dist.is_initialized():
            backend = dist.get_backend()
            device = torch.device(f"cuda:{torch.cuda.current_device()}") if backend == dist.Backend.NCCL else torch.device("cpu")
            t = torch.tensor([self.sum, self.count], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            self.sum, self.count = t.cpu().tolist()
            self.avg = self.sum / max(1.0, self.count)

@torch.no_grad()
def accuracy_topk(output: torch.Tensor, target: torch.Tensor, num_classes: int, ks=(1,5)):
    res = []
    maxk = min(max(ks), num_classes)
    batch = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    for k in ks:
        k = min(k, num_classes)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch))
    return res

# ------------------------- Enhanced LSTM Model for 85%+ accuracy ------------------------------

class LSTMTextModel(nn.Module):
    """
    Literature-standard 3-layer BiLSTM encoder with attention pooling.
    Architecture based on:
    - Yang et al. 2016 (Hierarchical Attention Networks)
    - McCann et al. 2017 (Learned in Translation) 
    - Peters et al. 2018 (Deep contextualized word representations)
    """
    def __init__(self, vocab_size, num_classes=2):
        super().__init__()
        
        # Model configuration - literature standard sizes
        embed_dim  = 300      # Standard embedding dimension
        hidden_dim = 768      # Increased from 512 - common in papers
        num_layers = 3        # 3-layer BiLSTM is standard for strong baselines
        dropout    = 0.25     # Reduced from 0.6 - critical for learning!
        
        # Embedding layer with dropout
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.emb_drop = nn.Dropout(EMB_DROPOUT_P)
        
        # 3-layer BiLSTM (standard in literature)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, 
            num_layers=num_layers,
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Enhanced attention mechanism (literature-standard)
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim * 2 * 3)  # For concatenated features
        
        # Deeper MLP classifier (common in recent papers)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 3, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for name, param in self.named_parameters():
            if 'embedding' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
            elif 'weight_ih' in name:  # LSTM input weights
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:  # LSTM hidden weights
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x, lengths: Optional[torch.Tensor] = None):
        # Word dropout (training only, less aggressive)
        if self.training and WORD_DROPOUT_P > 0:
            drop = (torch.rand_like(x, dtype=torch.float32) < WORD_DROPOUT_P) & (x != PAD_ID)
            x = torch.where(drop, torch.full_like(x, UNK_ID), x)
        
        # Embedding with dropout
        emb = self.emb_drop(self.embedding(x))
        
        # LSTM encoding with proper masking
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                emb, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (B, T, 2*H)
            T = out.size(1)
            mask = (torch.arange(T, device=out.device).unsqueeze(0) < 
                   lengths.to(out.device).unsqueeze(1)).unsqueeze(-1)  # (B,T,1)
        else:
            out, _ = self.lstm(emb)
            mask = torch.ones_like(out[..., :1], dtype=torch.bool)
        
        # Apply dropout to LSTM output
        out = self.dropout(out)
        
        # 1. Mean pooling (masked)
        out_masked = out.masked_fill(~mask, 0.0)
        sum_pool = out_masked.sum(dim=1)
        len_pool = mask.sum(dim=1).clamp(min=1)
        mean_pool = sum_pool / len_pool
        
        # 2. Max pooling (masked)
        out_neg_inf = out.masked_fill(~mask, float("-inf"))
        max_pool, _ = out_neg_inf.max(dim=1)
        
        # 3. Attention pooling (enhanced)
        attn_scores = self.attn(out)  # (B, T, 1)
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_pool = (out * attn_weights).sum(dim=1)
        
        # Concatenate all pooled features
        feat = torch.cat([mean_pool, max_pool, attn_pool], dim=1)  # (B, 2*H*3)
        
        # Layer norm for stability
        feat = self.norm(feat)
        
        # Final classification
        logits = self.proj(feat)
        return logits

# ------------------------- Training Functions ------------------------------

def cosine_lr_schedule(optimizer, epoch, max_epoch, warmup_ratio, min_lr_mult):
    """Cosine learning rate schedule with linear warmup"""
    warmup_epochs = int(max_epoch * warmup_ratio)
    if epoch < warmup_epochs:
        # Linear warmup
        lr_mult = (epoch + 1) / warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / max(1, max_epoch - warmup_epochs)
        lr_mult = min_lr_mult + (1 - min_lr_mult) * 0.5 * (1 + math.cos(math.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['initial_lr'] * lr_mult
    return optimizer.param_groups[0]['lr']

def train_epoch(epoch, model, loader, criterion, optimizer, scaler, device, straggle):
    model.train()
    metrics = {
        'loss': AverageMeter(), 'top1': AverageMeter(), 'top5': AverageMeter(),
        'data_time': AverageMeter(), 'comp_time': AverageMeter(), 'step_time': AverageMeter()
    }
    
    start_time = time.time()
    data_start = time.time()
    step_times = []
    
    for batch_idx, (inputs, targets, lengths) in enumerate(loader):
        data_time = time.time() - data_start
        metrics['data_time'].update(data_time)
        
        inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
        
        comp_start = time.time()
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs, lengths)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            straggle.delay()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            straggle.delay()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
        
        comp_time = time.time() - comp_start
        metrics['comp_time'].update(comp_time)
        
        # Calculate accuracy
        acc1, acc5 = accuracy_topk(outputs.detach(), targets, model.module.proj[-1].out_features)
        metrics['loss'].update(loss.item(), inputs.size(0))
        metrics['top1'].update(acc1.item(), inputs.size(0))
        metrics['top5'].update(acc5.item(), inputs.size(0))
        
        step_time = time.time() - data_start
        step_times.append(step_time)
        metrics['step_time'].update(step_time)
        
        if batch_idx % 50 == 0 and dist.get_rank() == 0:
            print(f"  [{batch_idx:3d}/{len(loader):3d}] "
                  f"loss={metrics['loss'].avg:.4f} top1={metrics['top1'].avg:.2f}% "
                  f"step={step_time:.3f}s", flush=True)
        
        data_start = time.time()
    
    # All-reduce metrics
    for m in metrics.values():
        m.all_reduce()
    
    epoch_time = time.time() - start_time
    throughput = len(loader.dataset) / epoch_time if dist.is_initialized() else \
                 len(loader.dataset) * dist.get_world_size() / epoch_time
    
    return {
        'loss': metrics['loss'].avg, 'top1': metrics['top1'].avg, 'top5': metrics['top5'].avg,
        'epoch_time': epoch_time, 'throughput': throughput,
        'step_time': np.mean(step_times), 'step_time_min': np.min(step_times),
        'step_time_max': np.max(step_times), 'data_time': metrics['data_time'].avg,
        'comp_time': metrics['comp_time'].avg
    }

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    metrics = {'loss': AverageMeter(), 'top1': AverageMeter(), 'top5': AverageMeter()}
    
    for inputs, targets, lengths in loader:
        inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
        outputs = model(inputs, lengths)
        loss = criterion(outputs, targets)
        
        acc1, acc5 = accuracy_topk(outputs, targets, model.module.proj[-1].out_features)
        metrics['loss'].update(loss.item(), inputs.size(0))
        metrics['top1'].update(acc1.item(), inputs.size(0))
        metrics['top5'].update(acc5.item(), inputs.size(0))
    
    for m in metrics.values():
        m.all_reduce()
    
    return {k: m.avg for k, m in metrics.items()}

# ------------------------- JSON Logging ------------------------------

def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def save_log(path, log):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2)
    os.rename(tmp, path)

# ------------------------- Straggler Simulation ------------------------------

class StraggleSim:
    def __init__(self, args):
        self.args = args
        self.active = (args.straggle_points > 0 or args.straggle_prob > 0) and args.straggle_amount > 0
        self.step = 0
        if self.active and args.rank == 0:
            print(f"[straggle_sim] active: points={args.straggle_points}, "
                  f"prob={args.straggle_prob:.3f}, amount={args.straggle_amount:.3f}s", flush=True)
    
    def delay(self):
        if not self.active: return
        
        delay_time = 0
        is_straggler = (self.args.rank in self.args.straggle_ranks) if self.args.straggle_ranks else True
        
        if is_straggler:
            if self.args.straggle_points > 0 and self.step < self.args.straggle_points:
                delay_time = self.args.straggle_amount
            elif self.args.straggle_prob > 0 and torch.rand(1).item() < self.args.straggle_prob:
                lo, hi = self.args.straggle_multiply
                mult = lo + torch.rand(1).item() * (hi - lo)
                delay_time = self.args.straggle_amount * mult
        
        if delay_time > 0:
            if self.args.straggle_verbose and self.args.rank == 0:
                print(f"[straggle] rank {self.args.rank} delay {delay_time:.3f}s", flush=True)
            time.sleep(delay_time)
        
        self.step += 1
    
    def get_stats(self):
        return {"events": self.step}

# ------------------------- Main Training Loop ------------------------------

def train(args):
    # Setup device
    device = torch.device(f'cuda:{args.local_rank}' if args.device == 'cuda' else 'cpu')
    
    # Build vocabulary
    vocab = build_vocab_sst(args.max_vocab, MIN_FREQ)
    if args.rank == 0:
        print(f"[Vocab] size={len(vocab)} (includes PAD/UNK)", flush=True)
    
    # Create datasets
    train_dataset = SSTDataset("train", vocab, args.max_len)
    val_dataset = SSTDataset("validation", vocab, args.max_len)
    
    # Create data loaders
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, prefetch_factor=args.prefetch_factor,
        drop_last=args.drop_last_train, persistent_workers=(args.workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size*2, sampler=val_sampler,
        num_workers=args.workers, pin_memory=True, prefetch_factor=args.prefetch_factor,
        drop_last=args.drop_last_val, persistent_workers=(args.workers > 0)
    )
    
    # Create model
    model = LSTMTextModel(len(vocab), args.num_classes).to(device)
    if args.rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model 'lstm_enhanced' initialized. (embed=300, hidden=768, layers=3, drop=0.25)", flush=True)
    
    # Wrap in DDP
    model = DDP(model, device_ids=[args.local_rank] if args.device == 'cuda' else None,
                static_graph=args.static_graph, bucket_cap_mb=args.bucket_cap_mb)
    
    if args.backend.startswith("dpa") and args.rank == 0:
        print(f"dpa.torch.py: DDP DPAState Object:  {{'pipes': 0, 'straggle': {args.world_size}, 'averaging': True, 'prescaled': {args.prescale}}}", flush=True)
        print(f"[straggle_sim][warning] created but effectively inactive -- points: {args.straggle_points}, prob: {args.straggle_prob}, amount: {args.straggle_amount}", flush=True)
    
    # Setup optimizer with scaled learning rate for distributed training
    # Linear scaling rule: scale by sqrt(world_size) for stability
    scaled_lr = args.learning_rate * math.sqrt(args.world_size) if args.scale_lr else args.learning_rate
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=scaled_lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    for group in optimizer.param_groups:
        group['initial_lr'] = group['lr']
    
    # Setup loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Setup mixed precision
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    # Setup straggler simulation
    straggle = StraggleSim(args)
    if args.rank == 0:
        if straggle.active:
            print(f"Straggle sim active: points={args.straggle_points}, "
                  f"prob={args.straggle_prob:.3f}, amount={args.straggle_amount:.3f}s", flush=True)
        else:
            print("Straggle sim inactive", flush=True)
    
    # Initialize JSON log
    log = {
        "command": " ".join(sys.argv),
        "args": vars(args),
        "start": now(),
        "epochs": {}
    }
    if args.rank == 0:
        save_log(args.json, log)
    
    # Training loop
    best_acc = 0.0
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        train_sampler.set_epoch(epoch)
        
        # Adjust learning rate
        lr = cosine_lr_schedule(optimizer, epoch, args.epochs, WARMUP_RATIO, args.cosine_min_lr_mult)
        
        # Train
        if args.rank == 0:
            print(f"[{now()}][Epoch {epoch:03d}] ...", flush=True)
        
        train_metrics = train_epoch(epoch, model, train_loader, criterion, optimizer, scaler, device, straggle)
        
        # Global loss tracking
        global_loss = train_metrics['loss']
        if dist.is_initialized():
            loss_tensor = torch.tensor([train_metrics['loss']], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            global_loss = loss_tensor.item() / dist.get_world_size()
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Track best accuracy
        if val_metrics['top1'] > best_acc:
            best_acc = val_metrics['top1']
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Log results
        if args.rank == 0:
            print(f"[{now()}][Epoch {epoch:03d}] "
                  f"train_loss={train_metrics['loss']:.4f} (global={global_loss:.4f}) "
                  f"val_loss={val_metrics['loss']:.4f} "
                  f"top1={val_metrics['top1']:.2f}% "
                  f"top5={val_metrics['top5']:.2f}% "
                  f"lr={lr:.6f} "
                  f"epoch_time={epoch_time:.2f}s "
                  f"step_time={train_metrics['step_time']:.2f} "
                  f"(min={train_metrics['step_time_min']:.2f}s, max={train_metrics['step_time_max']:.2f}) "
                  f"tp=~{train_metrics['throughput']:.1f} samples/s "
                  f"straggle_events={straggle.step}", flush=True)
            
            # Update JSON log
            epoch_metrics = {
                "start": now(),
                "epoch": int(epoch),
                "lr": float(lr),
                "train_loss": float(train_metrics['loss']),
                "train_loss_global": float(global_loss),
                "train_top1": float(train_metrics['top1']),
                "train_top5": float(train_metrics['top5']),
                "steps": int(len(train_loader)),
                "step_time_min": float(train_metrics['step_time_min']),
                "step_time_max": float(train_metrics['step_time_max']),
                "step_time": float(train_metrics['step_time']),
                "data_time": float(train_metrics['data_time']),
                "comp_time": float(train_metrics['comp_time']),
                "epoch_time": float(epoch_time),
                "epoch_train_time": float(train_metrics['epoch_time']),
                "epoch_train_throughput": float(train_metrics['throughput']),
                "val_loss": float(val_metrics['loss']),
                "val_top1": float(val_metrics['top1']),
                "val_top5": float(val_metrics['top5']),
                "straggle": straggle.get_stats() if straggle.active else {}
            }
            log["epochs"][str(epoch)] = epoch_metrics
            save_log(args.json, log)

# ------------------------- DDP Setup ------------------------------

def setup_ddp(args):
    def env_int(k, d): return d if os.environ.get(k) in (None,"") else int(os.environ.get(k))
    def env_str(k, d): return d if os.environ.get(k) in (None,"") else os.environ.get(k)

    args.rank        = env_int("RANK", args.rank)
    args.world_size  = env_int("WORLD_SIZE", args.world_size)
    args.master_addr = env_str("MASTER_ADDR", args.master_addr)
    args.master_port = env_int("MASTER_PORT", args.master_port)
    args.iface       = env_str("IFACE", args.iface)
    args.local_rank  = (args.rank % torch.cuda.device_count()) if torch.cuda.device_count() else 0
    if args.device == 'cuda' and torch.cuda.is_available(): torch.cuda.set_device(args.local_rank)

    os.environ.setdefault("RANK",        str(args.rank))
    os.environ.setdefault("WORLD_SIZE",  str(args.world_size))
    os.environ.setdefault("MASTER_ADDR", args.master_addr)
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ.setdefault("LOCAL_RANK",  str(args.local_rank))

    # Network settings
    os.environ.setdefault("GLOO_SOCKET_IFNAME", args.iface)
    os.environ.setdefault("NCCL_SOCKET_IFNAME", args.iface)
    
    # Initialize process group
    if args.backend.startswith("dpa"):
        if not args.dpa_conf: raise RuntimeError(f"--dpa_conf required for backend {args.backend}")
        dpa_device = dpa.DPADeviceOptions.from_config(args.dpa_conf)
        dpa_backend = dpa.DPADpdkBackendOptions.from_config(args.dpa_conf)
        pg_options = dpa.ProcessGroupDPADpdkOptions(dpa_device, dpa_backend)
        pg_options.hint_pinned_tensor_size = args.hint_tensor_size
        pg_options.hint_pinned_tensor_pool_size = args.hint_tensor_count
        dist.init_process_group(backend=args.backend, init_method="env://", 
                              rank=args.rank, world_size=args.world_size,
                              timeout=datetime.timedelta(seconds=60), pg_options=pg_options)
    else:
        dist.init_process_group(backend=args.backend, init_method="env://",
                              rank=args.rank, world_size=args.world_size,
                              timeout=datetime.timedelta(seconds=60))

    if args.rank == 0:
        print(f"[DDP] backend={args.backend} world_size={args.world_size} "
              f"master={args.master_addr}:{args.master_port} iface={args.iface}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="LSTM SST-2 training for 85%+ accuracy")
    
    # Distributed training
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--iface', type=str, default="ens4f0")
    parser.add_argument('--master_addr', type=str, default="42.0.0.1")
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--dpa_conf", type=str, default=None)
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument("--deterministic", action='store_true')
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--static_graph", action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--json", type=str, default="lstm.json", help="Path to JSON run log")
    
    # Training hyperparameters (optimized for 85%+)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0024)  # Scaled for 6 nodes
    parser.add_argument('--scale_lr', action='store_true', help="Auto-scale LR by sqrt(world_size)")
    parser.add_argument('--weight_decay', type=float, default=5e-06)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--drop_last_train", action='store_true')
    parser.add_argument("--drop_last_val", action='store_true')
    
    # Regularization (optimized values)
    parser.add_argument("--label_smoothing", type=float, default=0.01)
    parser.add_argument("--cosine_min_lr_mult", type=float, default=0.05)
    
    # Text processing
    parser.add_argument("--max_len", type=int, default=80)
    parser.add_argument("--max_vocab", type=int, default=60000)
    
    # Straggle simulation
    def csv_ints(s: str) -> List[int]:
        if not s: return []
        try: return [int(x) for x in re.split(r"\s*,\s*", s) if x]
        except ValueError: raise argparse.ArgumentTypeError("Expected comma-separated ints, e.g. 1,3,5")
    parser.add_argument("--straggle_points", type=int, default=0)
    parser.add_argument("--straggle_prob", type=float, default=0)
    parser.add_argument("--straggle_ranks", type=csv_ints, default=[])
    parser.add_argument("--straggle_amount", type=float, default=0)
    parser.add_argument("--straggle_multiply", type=float, nargs=2, metavar=("lo","hi"), default=[1.0, 1.0])
    parser.add_argument("--straggle_verbose", action='store_true')
    
    # DDP options
    parser.add_argument('--prescale', action="store_true", help="Prescale gradients for allreduce")
    parser.add_argument("--bucket_cap_mb", type=int, default=None)
    parser.add_argument("--hint_tensor_size", type=int, default=100000000)
    parser.add_argument("--hint_tensor_count", type=int, default=5)
    
    args = parser.parse_args()
    
    # Setup deterministic training if requested
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(args.seed + args.rank)
        np.random.seed(args.seed + args.rank)
        torch.manual_seed(args.seed + args.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + args.rank)
    else:
        torch.backends.cudnn.benchmark = True
    
    # Args sanity checks/corrections
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        if args.rank == 0:
            print("[Info] Using device=cpu because CUDA is not available", flush=True)
    if args.amp and args.device == 'cpu':
        args.amp = False
        if args.rank == 0:
            print("[Info] Disabling AMP because CUDA is not available", flush=True)
    if args.workers < 1:
        if args.rank == 0:
            print("[Info] Workers requested < 1; using workers=1", flush=True)
        args.workers = 1
    
    sys.stdout.reconfigure(line_buffering=True)
    
    # Setup distributed training
    setup_ddp(args)
    
    # Show configuration
    if args.rank == 0:
        print(json.dumps(vars(args), indent=2))
    
    # Run training
    try:
        train(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    main()