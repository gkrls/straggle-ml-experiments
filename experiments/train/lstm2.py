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

# ------------------------- Fixed NLP defaults (no external downloads) -------------------------
PAD_ID         = 0
UNK_ID         = 1
MIN_FREQ       = 1
WORD_DROPOUT_P = 0.0  # DISABLED for determinism
EMB_DROPOUT_P  = 0.1
CLIP_NORM      = 5.0
WARMUP_RATIO   = 0.10

# ------------------------- Tokenizer / Vocab ------------------------------
_token_re = re.compile(r"[a-z0-9']+|[!?.,;:()\-]+")

def simple_tokenizer(text: str) -> List[str]:
    text = text.lower().replace("\n", " ")
    text = re.sub(r"\d+", "<num>", text)
    return _token_re.findall(text)

def build_vocab_sst(max_vocab: Optional[int], min_freq: int) -> dict:
    from collections import Counter
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

# ------------------------- PROVEN LSTM MODEL FROM LITERATURE ------------------------------
class ProvenLSTM(nn.Module):
    """
    Standard BiLSTM for text classification - architecture proven to work on SST2
    Based on "Supervised Learning of Universal Sentence Representations" (2017)
    and standard BiLSTM baselines that achieve 85%+ on SST2
    """
    def __init__(self, vocab_size, num_classes=2):
        super().__init__()
        
        # Standard dimensions from literature
        embed_dim = 300
        hidden_dim = 512
        num_layers = 1  # Single layer BiLSTM often works best for SST2
        dropout = 0.5
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.embed_dropout = nn.Dropout(p=0.5)
        
        # BiLSTM 
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0 if num_layers == 1 else dropout
        )
        
        # Max pooling over time
        # Output projection
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
        # Initialize weights (important for stability)
        self.init_weights()
        
    def init_weights(self):
        # Initialize embeddings from uniform distribution
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[PAD_ID], 0)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                # Set forget gate bias to 1 (helps with gradient flow)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        # Initialize classifier
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        
    def forward(self, x, lengths=None):
        batch_size = x.size(0)
        
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = self.embed_dropout(embedded)
        
        # Pack sequences for LSTM
        if lengths is not None:
            # Sort by length for packing
            lengths_sorted, perm_idx = lengths.sort(0, descending=True)
            x_sorted = embedded[perm_idx]
            
            # Pack
            packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, lengths_sorted.cpu(), batch_first=True
            )
            
            # LSTM
            lstm_out, (hidden, cell) = self.lstm(packed)
            
            # Unpack
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
            
            # Unsort
            _, unsort_idx = perm_idx.sort(0)
            lstm_out = lstm_out[unsort_idx]
        else:
            lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Max pooling over time
        # Mask padding tokens
        if lengths is not None:
            mask = torch.arange(lstm_out.size(1), device=lstm_out.device).unsqueeze(0)
            mask = mask < lengths.unsqueeze(1)
            mask = mask.unsqueeze(2).expand_as(lstm_out)
            lstm_out = lstm_out.masked_fill(~mask, float('-inf'))
        
        # Max pool
        max_pooled, _ = torch.max(lstm_out, dim=1)  # (batch, hidden*2)
        
        # Handle case where entire sequence is padding
        max_pooled[max_pooled == float('-inf')] = 0
        
        # Classifier
        output = self.dropout(max_pooled)
        output = self.fc(output)
        
        return output

# ------------------------- Straggler Simulation ------------------------------
class Straggle:
    def __init__(self, args):
        self.active = False
        self.ranks_to_straggle = set()
        self.amounts = {}
        self.counts = {}
        
        if args.straggle_prob > 0 or len(args.straggle_ranks) > 0:
            self.active = True
            
            if len(args.straggle_ranks) > 0:
                self.ranks_to_straggle = set(args.straggle_ranks)
            elif args.straggle_points > 0:
                np.random.seed(args.seed)
                self.ranks_to_straggle = set(np.random.choice(
                    args.world_size, 
                    size=min(args.straggle_points, args.world_size),
                    replace=False
                ))
            
            for rank in range(args.world_size):
                self.counts[rank] = 0
                if rank in self.ranks_to_straggle:
                    if args.straggle_amount > 0:
                        self.amounts[rank] = args.straggle_amount
                    else:
                        np.random.seed(args.seed + rank)
                        mult = np.random.uniform(args.straggle_multiply[0], args.straggle_multiply[1])
                        self.amounts[rank] = mult
        
        self.prob = args.straggle_prob
        self.verbose = args.straggle_verbose
        self.rank = args.rank
    
    def delay(self, step_idx: int = 0):
        if not self.active:
            return
        
        if self.rank in self.ranks_to_straggle:
            np.random.seed(self.rank + step_idx)
            if np.random.random() < self.prob or self.prob == 0:
                delay_time = self.amounts[self.rank]
                if delay_time > 1.0:
                    base_sleep = 0.01
                    actual_delay = base_sleep * delay_time
                else:
                    actual_delay = delay_time
                
                if self.verbose and self.rank == 0:
                    print(f"[Straggle] Rank {self.rank} delaying {actual_delay:.3f}s", flush=True)
                
                time.sleep(actual_delay)
                self.counts[self.rank] += 1
    
    def get_stats(self):
        if not self.active:
            return {}
        
        return {
            "active_ranks": list(self.ranks_to_straggle),
            "straggle_counts": self.counts,
            "straggle_amounts": self.amounts
        }

# ------------------------- Training functions ------------------------------
def train_epoch(model, loader, optimizer, criterion, scaler, scheduler, device, epoch, args):
    model.train()
    
    metrics = {
        'loss': AverageMeter(),
        'top1': AverageMeter(),
        'top5': AverageMeter(),
        'data_time': AverageMeter(),
        'comp_time': AverageMeter(),
        'step_time': AverageMeter(),
        'step_time_min': float('inf'),
        'step_time_max': 0.0,
    }
    
    end = time.time()
    
    for batch_idx, (inputs, targets, lengths) in enumerate(loader):
        metrics['data_time'].update(time.time() - end)
        
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        
        comp_start = time.time()
        
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets)
            if args.prescale and args.world_size > 1:
                loss = loss / args.world_size
        
        optimizer.zero_grad(set_to_none=True)
        
        if args.amp:
            scaler.scale(loss).backward()
            # straggle.delay(epoch * len(loader) + batch_idx)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # straggle.delay(epoch * len(loader) + batch_idx)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
        
        scheduler.step()
        
        acc1, acc5 = accuracy_topk(outputs.detach(), targets, args.num_classes, ks=(1, 5))
        batch_size = inputs.size(0)
        metrics['loss'].update(loss.item() * (args.world_size if args.prescale and args.world_size > 1 else 1), batch_size)
        metrics['top1'].update(acc1.item(), batch_size)
        metrics['top5'].update(acc5.item(), batch_size)
        
        comp_time = time.time() - comp_start
        metrics['comp_time'].update(comp_time)
        
        step_time = time.time() - end
        metrics['step_time'].update(step_time)
        metrics['step_time_min'] = min(metrics['step_time_min'], step_time)
        metrics['step_time_max'] = max(metrics['step_time_max'], step_time)
        
        end = time.time()
    
    for key in ['loss', 'top1', 'top5', 'data_time', 'comp_time', 'step_time']:
        metrics[key].all_reduce()
    
    metrics['epoch_time'] = metrics['step_time'].sum
    metrics['throughput'] = len(loader.dataset) / metrics['epoch_time'] if metrics['epoch_time'] > 0 else 0
    
    return metrics

@torch.no_grad()
def validate(model, loader, criterion, device, args):
    model.eval()
    metrics = {'loss': AverageMeter(), 'top1': AverageMeter(), 'top5': AverageMeter()}
    
    for inputs, targets, lengths in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets)
        
        acc1, acc5 = accuracy_topk(outputs, targets, args.num_classes, ks=(1, 5))
        batch_size = inputs.size(0)
        
        metrics['loss'].update(loss.item(), batch_size)
        metrics['top1'].update(acc1.item(), batch_size)
        metrics['top5'].update(acc5.item(), batch_size)
    
    for m in metrics.values():
        m.all_reduce()
    
    return metrics

def save_log(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def train(args):
    log = {
        "args": vars(args),
        "epochs": {}
    }
    save_log(args.json, log)
    
    device = torch.device(f'cuda:{args.local_rank}' if args.device == 'cuda' else 'cpu')
    
    if args.rank == 0:
        print(f"[Setup] Building vocabulary (max_vocab={args.max_vocab}, min_freq={MIN_FREQ})...", flush=True)
    vocab = build_vocab_sst(args.max_vocab, MIN_FREQ)
    if args.rank == 0:
        print(f"[Setup] Vocabulary size: {len(vocab)}", flush=True)
    
    train_dataset = SSTDataset("train", vocab, args.max_len)
    val_dataset = SSTDataset("validation", vocab, args.max_len)
    
    if args.rank == 0:
        print(f"[Dataset] Train: {len(train_dataset)}, Val: {len(val_dataset)}", flush=True)
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=False,
        seed=args.seed
    )
    
    # Deterministic data loading
    def worker_init_fn(worker_id):
        worker_seed = args.seed + worker_id + args.rank * 1000
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    
    g = torch.Generator()
    g.manual_seed(args.seed + args.rank)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=args.drop_last_train,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        persistent_workers=args.workers > 0,
        worker_init_fn=worker_init_fn if args.deterministic else None,
        generator=g if args.deterministic else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=args.drop_last_val,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        persistent_workers=args.workers > 0,
        worker_init_fn=worker_init_fn if args.deterministic else None,
        generator=g if args.deterministic else None
    )
    
    if args.rank == 0:
        print(f"[DataLoader] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)
    
    # Model
    model = ProvenLSTM(len(vocab), args.num_classes).to(device)
    
    if args.rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Model] Total params: {total_params:,}, Trainable: {trainable_params:,}", flush=True)
    
    # DDP
    model = DDP(
        model,
        device_ids=[args.local_rank] if device.type == 'cuda' else None,
        # broadcast_buffers=False,
        bucket_cap_mb=args.bucket_cap_mb,
        gradient_as_bucket_view=True if args.static_graph else False,
        static_graph=args.static_graph
    )
    # Wrap the model if DPA backend is requested
    if args.backend.startswith("dpa"):
        model = dpa.DDPWrapper(model, straggle = args.world_size, prescale=args.prescale)

    # Straggle sim
    straggle = dpa.DDPStraggleSim(points=args.straggle_points, prob=args.straggle_prob, amount=args.straggle_amount, ranks=args.straggle_ranks,
                                  multiplier_range=args.straggle_multiply, verbose=args.straggle_verbose)      
    if straggle.attach(model): print(f"Straggle sim initialized with {straggle}")
    else: print(f"Straggle sim inactive")

    # Loss, Optimizer, Scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return max(args.cosine_min_lr_mult, 0.5 * (1 + math.cos(math.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    # straggle = Straggle(args)
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        
        epoch_start = time.time()
        
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, scaler, scheduler, device, epoch, args)
        val_metrics = validate(model, val_loader, criterion, device, args)
        
        epoch_time = time.time() - epoch_start
        
        if args.rank == 0:
            print(f"Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_metrics['loss'].avg:.4f} Top1: {train_metrics['top1'].avg:.2f}% Top5: {train_metrics['top5'].avg:.2f}% | "
                  f"Val Loss: {val_metrics['loss'].avg:.4f} Top1: {val_metrics['top1'].avg:.2f}% Top5: {val_metrics['top5'].avg:.2f}% | "
                  f"Time: {epoch_time:.1f}s", flush=True)
        
        # Log everything your original code logs
        epoch_metrics = {
            "train_loss": float(train_metrics['loss'].avg),
            "train_top1": float(train_metrics['top1'].avg),
            "train_top5": float(train_metrics['top5'].avg),
            "steps": int(len(train_loader)),
            "step_time_min": float(train_metrics['step_time_min']),
            "step_time_max": float(train_metrics['step_time_max']),
            "step_time": float(train_metrics['step_time'].avg),
            "data_time": float(train_metrics['data_time'].avg),
            "comp_time": float(train_metrics['comp_time'].avg),
            "epoch_time": float(epoch_time),
            "epoch_train_time": float(train_metrics['epoch_time']),
            "epoch_train_throughput": float(train_metrics['throughput']),
            "val_loss": float(val_metrics['loss'].avg),
            "val_top1": float(val_metrics['top1'].avg),
            "val_top5": float(val_metrics['top5'].avg),
            "straggle": straggle.get_stats() if straggle.active else {}
        }
        log["epochs"][str(epoch)] = epoch_metrics
        save_log(args.json, log)

# ------------------------- DDP Setup / Main ------------------------------
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

    os.environ.setdefault("GLOO_SOCKET_IFNAME", args.iface)
    os.environ.setdefault("GLOO_SOCKET_NTHREADS", "8")
    os.environ.setdefault("GLOO_NSOCKS_PERTHREAD", "2")
    os.environ.setdefault("GLOO_BUFFSIZE", "8388608")

    os.environ.setdefault("NCCL_SOCKET_IFNAME", args.iface)
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,NET,ENV")
    os.environ.setdefault("NCCL_DEBUG_FILE", f"/tmp/nccl_%h_rank{os.environ.get('RANK','0')}.log")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    os.environ.setdefault("NCCL_TREE_THRESHOLD", "0")
    os.environ.setdefault("NCCL_IB_DISABLE", "0")
    os.environ.setdefault("NCCL_BUFFSIZE", "8388608")
    os.environ.setdefault("NCCL_SOCKET_NTHREADS", "4")
    os.environ.setdefault("NCCL_NSOCKS_PERTHREAD", "4")

    if args.backend.startswith("dpa"):
        if not args.dpa_conf: raise RuntimeError(f"--dpa_conf required for backend {args.backend}")
        dpa_device = dpa.DPADeviceOptions.from_config(args.dpa_conf)
        dpa_backend = dpa.DPADpdkBackendOptions.from_config(args.dpa_conf)
        pg_options = dpa.ProcessGroupDPADpdkOptions(dpa_device, dpa_backend)
        pg_options.hint_pinned_tensor_size = max(args.hint_tensor_size, args.bucket_cap_mb * (2 ** 20) * 4 if args.bucket_cap_mb is not None else 0)
        pg_options.hint_pinned_tensor_pool_size = args.hint_tensor_count
        dist.init_process_group(backend=args.backend, init_method="env://", rank=args.rank, world_size=args.world_size, timeout = datetime.timedelta(seconds=60), pg_options=pg_options)
    else:
        dist.init_process_group(backend=args.backend, init_method="env://", rank=args.rank, world_size=args.world_size, timeout=datetime.timedelta(seconds=60))

    if args.rank == 0:
        print(f"[DDP] backend={args.backend} world_size={args.world_size} "
              f"master={args.master_addr}:{args.master_port} iface={args.iface} local_rank={args.local_rank}", flush=True)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--iface', type=str, default="ens4f0")
    parser.add_argument('--master_addr', type=str, default="42.0.0.1")
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument("--backend", type=str, default="gloo", help="DDP backend (e.g., gloo, nccl)")
    parser.add_argument("--dpa_conf", type=str, default=None, help="Path to dpa config.json")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument("--deterministic", action='store_true')
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=8)
    parser.add_argument("--static_graph", action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--json", type=str, default="lstm.json", help="Path to JSON run log")

    # Training (only BIG model)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0012)
    parser.add_argument('--weight_decay', type=float, default=2e-05)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--drop_last_train", action='store_true')
    parser.add_argument("--drop_last_val", action='store_true')
    parser.add_argument("--label_smoothing", type=float, default=0.02, help="CrossEntropy label smoothing")
    parser.add_argument("--cosine_min_lr_mult", type=float, default=0.1,
                        help="Cosine LR floor as a fraction of base LR")
    # Text knobs
    parser.add_argument("--max_len", type=int, default=64, help="Max tokens per sample")
    parser.add_argument("--max_vocab", type=int, default=60000, help="Max vocab size; 0 for unlimited")

    # Straggle
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

    parser.add_argument('--prescale', action="store_true", help="Prescale gradients for allreduce")
    parser.add_argument("--bucket_cap_mb", type=int, default=None, help="DDP bucket capacity")
    parser.add_argument("--hint_tensor_size", type=int, default=200000000, help="Hint for allreduce tensor size (bytes)")
    parser.add_argument("--hint_tensor_count", type=int, default=20, help="Hint for number of tensors allreduced, per step")

    args = parser.parse_args()

    if args.deterministic:
        args.seed = args.seed + args.rank
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        g = torch.Generator()
        g.manual_seed(args.seed)
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

    setup_ddp(args)

    print(json.dumps(vars(args), indent=2))
    try:
        train(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    main()