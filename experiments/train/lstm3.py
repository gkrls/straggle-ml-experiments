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


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = [p.detach().clone() for p in model.parameters() if p.requires_grad]
        self.params = [p for p in model.parameters() if p.requires_grad]
    @torch.no_grad()
    def update(self):
        for s, p in zip(self.shadow, self.params):
            s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
    @torch.no_grad()
    def load_shadow(self, model):
        for s, p in zip(self.shadow, self.params):
            p.data.copy_(s.data)

def unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, DDP) else m

# ------------------------- Fixed NLP defaults (no external downloads) -------------------------
PAD_ID         = 0
UNK_ID         = 1
MIN_FREQ       = 1            # keep rare tokens (helps sentiment)
WORD_DROPOUT_P = 0.0 #0.05 #0.15         # bumped from 0.10
EMB_DROPOUT_P  = 0.1 #0.20         # bumped from 0.10
CLIP_NORM      = 5.0
WARMUP_RATIO   = 0.10         # 10% warmup for cosine

# ------------------------- Tokenizer / Vocab ------------------------------

_token_re = re.compile(r"[a-z0-9']+|[!?.,;:()\-]+")

def simple_tokenizer(text: str) -> List[str]:
    text = text.lower().replace("\n", " ")
    text = re.sub(r"\d+", "<num>", text)  # fold numbers
    return _token_re.findall(text)

def build_vocab_sst(max_vocab: Optional[int], min_freq: int) -> dict:
    from collections import Counter
    # build from train + validation to reduce OOV (no external data)
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

# ------------------------- Model (only BIG preset) ------------------------------

class LSTMTextModel(nn.Module):
    """
    2-layer BiLSTM encoder + pooled readout:
      - mean pool (masked)
      - max  pool (masked)
      - single-head self-attention pooling (fp16 safe)
    Final feature -> LayerNorm -> small MLP head.
    """
    def __init__(self, vocab_size, num_classes=2):
        super().__init__()
        embed_dim  = 300
        hidden_dim = 768
        num_layers = 2
        dropout    = 0.4

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.emb_drop = nn.Dropout(EMB_DROPOUT_P)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        # init LSTM nicely + forget gate bias = 1.0 (helps stability)
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.xavier_uniform_(p)
            elif "bias_ih" in name or "bias_hh" in name:
                nn.init.zeros_(p)
                # set forget gate bias (f gate is the 2nd quarter)
                q = p.shape[0] // 4
                p.data[q:2*q].fill_(1.0)

        self.attn = nn.Linear(hidden_dim * 2, 1, bias=False)   # attention scores
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim * 2 * 3)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2 * 3, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, lengths: Optional[torch.Tensor] = None):
        # --- word dropout (train-time only; keep PAD intact) ---
        if self.training and WORD_DROPOUT_P > 0:
            drop = (torch.rand_like(x, dtype=torch.float32) < WORD_DROPOUT_P) & (x != PAD_ID)
            x = torch.where(drop, torch.full_like(x, UNK_ID), x)

        emb = self.emb_drop(self.embedding(x))

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (B, T, 2H)
            T = out.size(1)
            mask = (torch.arange(T, device=out.device).unsqueeze(0) < lengths.to(out.device).unsqueeze(1)).unsqueeze(-1)  # (B,T,1)
        else:
            out, _ = self.lstm(emb)
            mask = torch.ones_like(out[..., :1], dtype=torch.bool)

        # mean pool (masked)
        out_masked = out.masked_fill(~mask, 0.0)
        sum_pool = out_masked.sum(dim=1)
        len_pool = mask.sum(dim=1).clamp(min=1)
        mean_pool = sum_pool / len_pool

        # max pool (masked)
        out_neg_inf = out.masked_fill(~mask, float("-inf"))
        max_pool, _ = out_neg_inf.max(dim=1)
        max_pool[max_pool == float("-inf")] = 0.0

        # attention pool (masked softmax), fp16-safe
        scores = self.attn(out).squeeze(-1)      # (B,T)
        mask_t = mask.squeeze(-1)                # (B,T)
        fill = torch.finfo(scores.dtype).min if scores.dtype != torch.float64 else -1e9
        scores = scores.masked_fill(~mask_t, fill)
        alpha  = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B,T,1)
        attn_pool = (out * alpha).sum(dim=1)     # (B,2H)

        feat = torch.cat([mean_pool, max_pool, attn_pool], dim=1)  # (B, 6H)
        feat = self.norm(feat)
        return self.proj(self.dropout(feat))

# ------------------------- Scheduler (per-step cosine+warmup, with floor) ------------------------------

def build_per_step_cosine(optimizer, total_steps: int, warmup_steps: int, min_lr_mult: float = 0.15):
    """
    Cosine schedule with warmup and a non-zero floor so LR never collapses.
    Effective LR = base_lr * lambda(step); lambda in [min_lr_mult, 1].
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_mult + (1.0 - min_lr_mult) * cos
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

# ------------------------- Train / Eval ------------------------------



# @torch.no_grad()
# def validate(model, loader, device, args, num_classes: int):
#     model.eval()
#     top1, top5, losses = AverageMeter(), AverageMeter(), AverageMeter()
#     criterion = nn.CrossEntropyLoss().to(device)

#     def run_validation(dataloader):
#         for x, y, lengths in dataloader:
#             x = x.to(device, non_blocking=True)
#             y = y.to(device, non_blocking=True)
#             out = model(x, lengths)   # lengths on CPU
#             loss = criterion(out, y)
#             a1, a5 = accuracy_topk(out, y, num_classes=num_classes, ks=(1,5))
#             bs = y.size(0)
#             losses.update(loss.item(), bs)
#             top1.update(a1[0].item(), bs)
#             top5.update(a5[0].item(), bs)

#     run_validation(loader)
#     top1.all_reduce(); top5.all_reduce(); losses.all_reduce()

#     # Leftover handling 
#     if hasattr(loader, "sampler") and len(loader.sampler) * args.world_size < len(loader.dataset):
#         start = len(loader.sampler) * args.world_size
#         aux = Subset(loader.dataset, range(start, len(loader.dataset)))
#         aux_loader = DataLoader(aux, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
#         run_validation(aux_loader)

#     return {'loss': losses.avg, 'top1': top1.avg, 'top5': top5.avg}
@torch.no_grad()
def validate(model, loader, device, args, num_classes: int):
    model.eval()
    top1, top5, losses = AverageMeter(), AverageMeter(), AverageMeter()
    criterion = nn.CrossEntropyLoss().to(device)

    for x, y, lengths in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x, lengths)   # lengths on CPU
        loss = criterion(out, y)
        a1, a5 = accuracy_topk(out, y, num_classes=num_classes, ks=(1,5))
        bs = y.size(0)
        losses.update(loss.item(), bs)
        top1.update(a1[0].item(), bs)
        top5.update(a5[0].item(), bs)

    top1.all_reduce(); top5.all_reduce(); losses.all_reduce()
    return {'loss': losses.avg, 'top1': top1.avg, 'top5': top5.avg}

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, scaler, num_classes: int,
                    ema: Optional[EMA] = None):
    model.train()
    losses = AverageMeter(); top1 = AverageMeter(); top5 = AverageMeter()
    step_time = AverageMeter(); data_time = AverageMeter()

    # epoch timing
    if device.type == 'cuda':
        e_start = torch.cuda.Event(enable_timing=True); e_end = torch.cuda.Event(enable_timing=True); e_start.record()
    else:
        e_wall = time.perf_counter()

    step_start = time.perf_counter()
    samples_seen = 0.0

    for x, y, lengths in dataloader:
        data_time.update(time.perf_counter() - step_start, 1)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        samples_seen += y.size(0)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                out = model(x, lengths)  # lengths stays on CPU
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            if CLIP_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x, lengths)
            loss = criterion(out, y)
            loss.backward()
            if CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)
            optimizer.step()

        # --- SWA update (no BN in this model, so this is enough)
        if ema is not None:
            ema.update()

        # per-step scheduler
        if scheduler: scheduler.step()

        # metrics
        a1, a5 = accuracy_topk(out, y, num_classes=num_classes, ks=(1,5))
        bs = y.size(0)
        losses.update(loss.item(), bs)
        top1.update(a1[0].item(), bs)
        top5.update(a5[0].item(), bs)

        step_time.update(time.perf_counter() - step_start, 1)
        step_start = time.perf_counter()

    # epoch duration
    if device.type == 'cuda':
        e_end.record(); e_end.synchronize()
        duration = e_start.elapsed_time(e_end) / 1000.0
    else:
        duration = time.perf_counter() - e_wall

    throughput = samples_seen / max(1e-6, duration)

    # global loss average
    t = torch.tensor([losses.sum, losses.count], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    loss_global = (t[0] / torch.clamp(t[1], min=1.0)).item()

    return {
        'loss_global' : loss_global,
        'loss': losses.avg,
        'top1': top1.avg,
        'top5': top5.avg,
        'step_time_min': step_time.min,
        'step_time_max': step_time.max,
        'step_time': step_time.avg,
        'data_time': data_time.avg,
        'comp_time': step_time.avg - data_time.avg,
        'epoch_time': duration,
        'throughput': throughput,   # samples/s
    }

# ------------------------- Logging ------------------------------

def save_log(path, log):
    """Atomically write log dict to JSON file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

# ------------------------- Dataloaders ------------------------------

def get_dataloaders(args, vocab: dict):
    train_ds = SSTDataset("train", vocab, max_len=args.max_len)
    val_ds   = SSTDataset("validation", vocab, max_len=args.max_len)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=args.world_size, rank=args.rank,
        shuffle=True, drop_last=args.drop_last_train
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=args.world_size, rank=args.rank,
        shuffle=False, drop_last=args.drop_last_val
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=args.prefetch_factor
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=True, prefetch_factor=args.prefetch_factor
    )
    return train_loader, val_loader

# ------------------------- Train Driver ------------------------------

def train(args):
    device = torch.device(args.device)

    # ------- vocab on rank-0 then broadcast -------
    if dist.get_rank() == 0:
        cap = None if args.max_vocab in (0, None) else args.max_vocab
        vocab = build_vocab_sst(max_vocab=cap, min_freq=MIN_FREQ)
    else:
        vocab = None
    obj = [vocab]; dist.broadcast_object_list(obj, src=0); vocab = obj[0]
    if dist.get_rank() == 0:
        print(f"[Vocab] size={len(vocab)} (includes PAD/UNK)")

    # Data
    train_loader, val_loader = get_dataloaders(args, vocab)

    # Model (only BIG)
    model = LSTMTextModel(vocab_size=len(vocab), num_classes=args.num_classes).to(device)
    model = DDP(model, device_ids=[args.local_rank] if device.type == "cuda" else None,
                gradient_as_bucket_view=True, find_unused_parameters=False, static_graph=args.static_graph)

    # Wrap the model if DPA backend is requested
    if args.backend.startswith("dpa"):
        model = dpa.DDPWrapper(model, straggle = args.world_size, prescale=args.prescale)
    
    unwrap(model).lstm.flatten_parameters()  # you already call this later; keeping it here is good too
    ema = EMA(unwrap(model), decay=0.999)

    # print(f"Model 'lstm_big' initialized. (embed=300, hidden=512, layers=2, drop=0.6)", flush=True)
    print(f"Model 'lstm_big' initialized. (embed=300, hidden=768, layers=2, drop=0.5)", flush=True)

    # Straggle sim
    straggle = dpa.DDPStraggleSim(points=args.straggle_points, prob=args.straggle_prob, amount=args.straggle_amount, ranks=args.straggle_ranks,
                                  multiplier_range=args.straggle_multiply, verbose=args.straggle_verbose)     
    if straggle.attach(model): print(f"Straggle sim initialized with {straggle}")
    else: print(f"Straggle sim inactive")

    # Loss / Optim / Scheduler / AMP
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,
                            weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = build_per_step_cosine(optimizer, total_steps, warmup_steps, min_lr_mult=args.cosine_min_lr_mult)

    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and device.type == "cuda"))

    def now(): return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # JSON log
    log = {"time": now(), "config": vars(args), "epochs": {}}
    save_log(args.json, log)

    for epoch in range(args.epochs):
        print(f"[{now()}][Epoch {epoch:03d}] ...")
        epoch_start = time.time()

        straggle.reset_stats()

        train_loader.sampler.set_epoch(epoch)
        unwrap(model).lstm.flatten_parameters()


        step_scheduler = scheduler

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, step_scheduler,
                                        device, scaler, num_classes=args.num_classes, ema=ema)
        
        eval_model = swa_model if swa_active else model

        raw_state = [p.detach().clone() for p in unwrap(model).parameters()]
        ema.load_shadow(unwrap(model))
        val_metrics = validate(model, val_loader, device, args, num_classes=args.num_classes)

        # train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler,
        #                                 num_classes=args.num_classes)
        # val_metrics   = validate(model, val_loader, device, args, num_classes=args.num_classes)

        epoch_time = time.time() - epoch_start

        current_lr = optimizer.param_groups[0]['lr']

        print(f"[{now()}][Epoch {epoch:03d}] "
              f"train_loss={train_metrics['loss']:.4f} (global={train_metrics['loss_global']:.4f}) "
              f"val_loss={val_metrics['loss']:.4f} "
              f"top1={val_metrics['top1']:.2f}% top5={val_metrics['top5']:.2f}% "
              f"lr={current_lr:.6f} epoch_time={epoch_time:.2f}s step_time={train_metrics['step_time']:.2f} "
              f"(min={train_metrics['step_time_min']:.2f}s, max={train_metrics['step_time_max']:.2f}) "
              f"tp=~{train_metrics['throughput']:.1f} samples/s",
              f"straggle_events={straggle.get_stats()['num_straggle_events']}",
              flush=True)

        # JSON epoch record
        epoch_metrics = {
            "lr": float(current_lr),
            "train_loss": float(train_metrics['loss']),
            "train_loss_global": float(train_metrics['loss_global']),
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
        # pg_options.hint_pinned_tensor_size = max(200_000_000, args.bucket_cap_mb * (2 ** 20) * 4) # observed max around 150-is MB
        # pg_options.hint_pinned_tensor_pool_size = 20                                              # observed count 13
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
    parser.add_argument('--learning_rate', type=float, default=0.0012) # 0.001
    parser.add_argument('--weight_decay', type=float, default=1e-05) # 1e-05
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--drop_last_train", action='store_true')
    parser.add_argument("--drop_last_val", action='store_true')
    parser.add_argument("--label_smoothing", type=float, default=0.00, help="CrossEntropy label smoothing") # 0.05
    parser.add_argument("--cosine_min_lr_mult", type=float, default=0.0,
                        help="Cosine LR floor as a fraction of base LR (e.g., 0.15 keeps LR >= 15% of base)") # 0.15
    # Text knobs
    parser.add_argument("--max_len", type=int, default=96, help="Max tokens per sample") # 50
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
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(args.seed + args.rank)
        np.random.seed(args.seed + args.rank)
        torch.manual_seed(args.seed + args.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + args.seed)
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