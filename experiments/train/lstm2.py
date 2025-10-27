#!/usr/bin/env python3
"""
Deterministic LSTM for SST2 - Guaranteed reproducible results
"""

import argparse
import os
import sys
import json
import time
import datetime
import random
import numpy as np
from typing import Optional, List
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from datasets import load_dataset

# DPA import if available
try:
    import dpa
    HAS_DPA = True
except ImportError:
    HAS_DPA = False

# ======================== FIXED CONSTANTS ========================
PAD_ID = 0
UNK_ID = 1
MAX_LEN = 64
VOCAB_SIZE = 20000
EMB_DIM = 300
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.3
LABEL_SMOOTHING = 0.05

# ======================== DETERMINISTIC SEED SETUP ========================
def seed_everything(seed, rank=0):
    """Completely deterministic seeding"""
    seed = seed + rank  # Different seed per rank
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

# ======================== TOKENIZER & VOCAB ========================
def simple_tokenize(text):
    """Simple deterministic tokenizer"""
    import re
    text = text.lower()
    tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
    return tokens

def build_vocab(max_vocab=VOCAB_SIZE):
    """Build vocabulary from training data"""
    from collections import Counter
    ds_train = load_dataset("glue", "sst2", split="train")
    counter = Counter()
    
    for ex in ds_train:
        tokens = simple_tokenize(ex["sentence"])
        counter.update(tokens)
    
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    most_common = counter.most_common(max_vocab - 2)
    
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    return vocab

def encode_text(text, vocab, max_len=MAX_LEN):
    """Encode text to indices"""
    tokens = simple_tokenize(text)[:max_len]
    ids = [vocab.get(tok, UNK_ID) for tok in tokens]
    length = len(ids)
    
    # Pad to max_len
    if length < max_len:
        ids += [PAD_ID] * (max_len - length)
    
    return torch.tensor(ids, dtype=torch.long), length

# ======================== DATASET ========================
class SSTDataset(Dataset):
    def __init__(self, split, vocab, max_len=MAX_LEN):
        self.data = load_dataset("glue", "sst2", split=split)
        self.vocab = vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text_ids, length = encode_text(item["sentence"], self.vocab, self.max_len)
        label = torch.tensor(item["label"], dtype=torch.long)
        return text_ids, label, length

# ======================== LSTM MODEL ========================
class DeterministicLSTM(nn.Module):
    """Deterministic LSTM for text classification"""
    
    def __init__(self, vocab_size, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMB_DIM, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(DROPOUT)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            EMB_DIM, HIDDEN_DIM, 
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, num_classes)
        )
        
        # Initialize weights deterministically
        self._init_weights()
    
    def _init_weights(self):
        """Deterministic weight initialization"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'embedding' in name:
                nn.init.normal_(param, mean=0, std=0.1)
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x, lengths=None):
        # Embedding and dropout
        embedded = self.dropout(self.embedding(x))
        
        # Pack sequences for efficiency
        if lengths is not None:
            lengths = lengths.cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            lstm_out, (hidden, _) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (hidden, _) = self.lstm(embedded)
        
        # Use last hidden state from both directions
        # hidden shape: (num_layers * num_directions, batch, hidden_dim)
        forward_hidden = hidden[-2, :, :]  # Last layer forward
        backward_hidden = hidden[-1, :, :]  # Last layer backward
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Classification
        output = self.fc(combined)
        return output

# ======================== METRICS ========================
class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def all_reduce(self):
        if dist.is_available() and dist.is_initialized():
            t = torch.tensor([self.sum, self.count], dtype=torch.float64)
            if torch.cuda.is_available():
                t = t.cuda()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            self.sum, self.count = t.cpu().tolist()
            self.avg = self.sum / max(1, self.count)

# ======================== TRAINING ========================
def train_epoch(model, loader, optimizer, criterion, device, rank):
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    for batch_idx, (texts, labels, lengths) in enumerate(loader):
        texts = texts.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        
        # Forward pass
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        
        optimizer.step()
        
        # Metrics
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        
        loss_meter.update(loss.item(), texts.size(0))
        acc_meter.update(acc.item(), texts.size(0))
    
    # Synchronize metrics across ranks
    loss_meter.all_reduce()
    acc_meter.all_reduce()
    
    return loss_meter.avg, acc_meter.avg

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    
    for texts, labels, lengths in loader:
        texts = texts.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        
        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)
        
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).float().mean()
        
        loss_meter.update(loss.item(), texts.size(0))
        acc_meter.update(acc.item(), texts.size(0))
    
    loss_meter.all_reduce()
    acc_meter.all_reduce()
    
    return loss_meter.avg, acc_meter.avg

# ======================== MAIN TRAINING LOOP ========================
def train(args):
    # Seed everything first
    seed = seed_everything(args.seed, args.rank)
    
    if args.rank == 0:
        print(f"[Setup] Seed={seed}, World Size={args.world_size}, Backend={args.backend}")
    
    # Device setup
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    
    # Build vocab and datasets
    vocab = build_vocab(VOCAB_SIZE)
    train_dataset = SSTDataset("train", vocab)
    val_dataset = SSTDataset("validation", vocab)
    
    # Create samplers
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=args.world_size, 
        rank=args.rank,
        shuffle=True,
        seed=args.seed  # Fixed seed for sampler
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=False
    )
    
    # DataLoader with deterministic worker init
    def worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id + args.rank * 100
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    # Create generator for DataLoader
    g = torch.Generator()
    g.manual_seed(args.seed + args.rank)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=min(4, args.workers),
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=g
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=min(4, args.workers),
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=g
    )
    
    # Model
    model = DeterministicLSTM(len(vocab), num_classes=2).to(device)
    
    # DDP wrapping
    if args.world_size > 1:
        model = DDP(
            model, 
            device_ids=[args.local_rank] if device.type == 'cuda' else None,
            broadcast_buffers=False,
            bucket_cap_mb=args.bucket_cap_mb
        )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    
    # Cosine scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.1
    )
    
    # Training loop
    best_val_acc = 0.0
    log_data = {"epochs": {}}
    
    for epoch in range(1, args.epochs + 1):
        # CRITICAL: Set epoch for sampler to ensure different shuffling each epoch
        train_sampler.set_epoch(epoch)
        
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, args.rank)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Logging
        if args.rank == 0:
            print(f"Epoch {epoch}/{args.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, "
                  f"Time: {epoch_time:.2f}s")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'vocab': vocab,
                }, 'best_lstm_model.pt')
                print(f"  → New best model saved! Acc: {val_acc*100:.2f}%")
            
            # Log to JSON
            log_data["epochs"][str(epoch)] = {
                "train_loss": float(train_loss),
                "train_acc": float(train_acc * 100),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc * 100),
                "epoch_time": float(epoch_time),
                "lr": optimizer.param_groups[0]['lr']
            }
            
            with open(args.json, 'w') as f:
                json.dump(log_data, f, indent=2)
    
    if args.rank == 0:
        print(f"\n✓ Training complete! Best validation accuracy: {best_val_acc*100:.2f}%")

# ======================== DDP SETUP ========================
def setup_ddp(args):
    """Setup DDP environment"""
    def env_int(k, d): 
        return d if os.environ.get(k) in (None, "") else int(os.environ.get(k))
    
    def env_str(k, d): 
        return d if os.environ.get(k) in (None, "") else os.environ.get(k)
    
    args.rank = env_int("RANK", args.rank)
    args.world_size = env_int("WORLD_SIZE", args.world_size)
    args.master_addr = env_str("MASTER_ADDR", args.master_addr)
    args.master_port = env_int("MASTER_PORT", args.master_port)
    args.iface = env_str("IFACE", args.iface)
    args.local_rank = (args.rank % torch.cuda.device_count()) if torch.cuda.is_available() else 0
    
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    
    # Set environment variables
    os.environ.setdefault("RANK", str(args.rank))
    os.environ.setdefault("WORLD_SIZE", str(args.world_size))
    os.environ.setdefault("MASTER_ADDR", args.master_addr)
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    
    # Backend-specific settings
    if args.backend == "gloo":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", args.iface)
    elif args.backend == "nccl":
        os.environ.setdefault("NCCL_SOCKET_IFNAME", args.iface)
        os.environ.setdefault("NCCL_DEBUG", "WARN")
    
    # Initialize DDP
    if args.backend.startswith("dpa") and HAS_DPA:
        # DPA backend setup
        dpa_device = dpa.DPADeviceOptions.from_config(args.dpa_conf)
        dpa_backend = dpa.DPADpdkBackendOptions.from_config(args.dpa_conf)
        pg_options = dpa.ProcessGroupDPADpdkOptions(dpa_device, dpa_backend)
        pg_options.hint_pinned_tensor_size = args.hint_tensor_size
        pg_options.hint_pinned_tensor_pool_size = args.hint_tensor_count
        
        dist.init_process_group(
            backend=args.backend,
            init_method="env://",
            rank=args.rank,
            world_size=args.world_size,
            timeout=datetime.timedelta(seconds=60),
            pg_options=pg_options
        )
    else:
        dist.init_process_group(
            backend=args.backend,
            init_method="env://",
            rank=args.rank,
            world_size=args.world_size,
            timeout=datetime.timedelta(seconds=60)
        )

def main():
    parser = argparse.ArgumentParser(description="Deterministic LSTM for SST2")
    
    # DDP arguments
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--iface', type=str, default="ens4f0")
    parser.add_argument('--master_addr', type=str, default="42.0.0.1")
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument("--backend", type=str, default="gloo")
    parser.add_argument("--dpa_conf", type=str, default=None)
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument("--workers", type=int, default=4)
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument("--bucket_cap_mb", type=int, default=25)
    
    # DPA hints
    parser.add_argument("--hint_tensor_size", type=int, default=200000000)
    parser.add_argument("--hint_tensor_count", type=int, default=20)
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--json", type=str, default="lstm_deterministic.json")
    
    args = parser.parse_args()
    
    # Always use deterministic mode
    args.deterministic = True
    
    # Setup DDP
    setup_ddp(args)
    
    try:
        train(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    main()