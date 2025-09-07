import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset, DataLoader
import sys
import json
import datetime
import time
import re
import random
import numpy as np
import math
from typing import List, Dict

from straggle_sim import SlowWorkerPattern

# HF
from datasets import load_dataset, DownloadMode
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForQuestionAnswering,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
from torch.cuda.amp import GradScaler, autocast

# ------------------------- Dataset ------------------------------

def _ensure_data_root(root: str):
    """Single directory for everything. Internals are hidden inside it."""
    os.makedirs(root, exist_ok=True)
    ds_cache = os.path.join(root, ".hf_datasets")
    tf_cache = os.path.join(root, ".hf_transformers")
    os.makedirs(ds_cache, exist_ok=True)
    os.makedirs(tf_cache, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", ds_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", tf_cache)

def _prepare_features(args, raw, tokenizer):
    max_len = args.max_seq_len
    doc_stride = args.doc_stride

    def prep_train(ex):
        tok = tokenizer(
            [q.strip() for q in ex['question']], ex['context'],
            truncation='only_second', max_length=max_len, stride=doc_stride,
            return_overflowing_tokens=True, return_offsets_mapping=True,
            padding='max_length',
        )
        sample_map = tok.pop('overflow_to_sample_mapping')
        offsets = tok.pop('offset_mapping')
        start_positions, end_positions = [], []
        for i, offs in enumerate(offsets):
            input_ids = tok['input_ids'][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            si = sample_map[i]
            answers = ex['answers'][si]
            if len(answers['answer_start']) == 0:
                start_positions.append(cls_index); end_positions.append(cls_index); continue
            start_char = answers['answer_start'][0]
            end_char = start_char + len(answers['text'][0])
            seq_ids = tok.sequence_ids(i)
            k = 0
            while k < len(seq_ids) and seq_ids[k] != 1: k += 1
            ctx_start = k
            while k < len(seq_ids) and seq_ids[k] == 1: k += 1
            ctx_end = k - 1
            if not (offs[ctx_start][0] <= start_char and offs[ctx_end][1] >= end_char):
                start_positions.append(cls_index); end_positions.append(cls_index)
            else:
                s = ctx_start
                while s <= ctx_end and offs[s][0] <= start_char: s += 1
                e = ctx_end
                while e >= ctx_start and offs[e][1] >= end_char: e -= 1
                start_positions.append(s - 1)
                end_positions.append(e + 1)
        tok['start_positions'] = start_positions
        tok['end_positions'] = end_positions
        return tok

    def prep_val(ex):
        tok = tokenizer(
            [q.strip() for q in ex['question']], ex['context'],
            truncation='only_second', max_length=max_len, stride=doc_stride,
            return_overflowing_tokens=True, return_offsets_mapping=True,
            padding='max_length',
        )
        sample_map = tok.pop('overflow_to_sample_mapping')
        offsets = tok.pop('offset_mapping')
        # labels for val loss + metadata for EM/F1
        start_positions, end_positions = [], []
        example_id, cleaned_offsets = [], []
        for i, offs in enumerate(offsets):
            input_ids = tok['input_ids'][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            si = sample_map[i]
            example_id.append(ex['id'][si])
            ans = ex['answers'][si]
            # labels (use first answer if exists, else CLS)
            if len(ans['answer_start']) == 0:
                start_positions.append(cls_index); end_positions.append(cls_index)
            else:
                start_char = ans['answer_start'][0]
                end_char = start_char + len(ans['text'][0])
                seq_ids = tok.sequence_ids(i)
                k = 0
                while k < len(seq_ids) and seq_ids[k] != 1: k += 1
                ctx_start = k
                while k < len(seq_ids) and seq_ids[k] == 1: k += 1
                ctx_end = k - 1
                if not (offs[ctx_start][0] <= start_char and offs[ctx_end][1] >= end_char):
                    start_positions.append(cls_index); end_positions.append(cls_index)
                else:
                    s = ctx_start
                    while s <= ctx_end and offs[s][0] <= start_char: s += 1
                    e = ctx_end
                    while e >= ctx_start and offs[e][1] >= end_char: e -= 1
                    start_positions.append(s - 1)
                    end_positions.append(e + 1)
            # cleaned offsets for extracting text
            seq_ids = tok.sequence_ids(i)
            cleaned = []
            for k, o in enumerate(offs):
                cleaned.append(o if seq_ids[k] == 1 else (0, 0))
            cleaned_offsets.append(cleaned)
        tok['start_positions'] = start_positions
        tok['end_positions'] = end_positions
        tok['example_id'] = example_id
        tok['offset_mapping'] = cleaned_offsets
        return tok

    train_features = raw['train'].map(prep_train, batched=True, remove_columns=raw['train'].column_names, desc='Tokenize train')
    val_features   = raw['validation'].map(prep_val, batched=True, remove_columns=raw['validation'].column_names, desc='Tokenize val')

    # For train, tensors; for val, keep Python objects and collate manually
    train_cols = ['input_ids', 'attention_mask', 'start_positions', 'end_positions']
    if 'token_type_ids' in train_features.column_names:
        train_cols.insert(2, 'token_type_ids')
    train_features.set_format(type='torch', columns=train_cols)
    val_features.set_format(type=None)
    return train_features, val_features

def _collate_val(batch: List[Dict[str, object]]):
    # tensors for inputs + labels; lists for metadata
    input_ids      = torch.tensor([b['input_ids'] for b in batch], dtype=torch.long)
    attention_mask = torch.tensor([b['attention_mask'] for b in batch], dtype=torch.long)
    token_type_ids = torch.tensor([b.get('token_type_ids', [0]*len(b['input_ids'])) for b in batch], dtype=torch.long)
    start_positions= torch.tensor([b['start_positions'] for b in batch], dtype=torch.long)
    end_positions  = torch.tensor([b['end_positions'] for b in batch], dtype=torch.long)
    example_id     = [b['example_id'] for b in batch]
    offset_mapping = [b['offset_mapping'] for b in batch]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'start_positions': start_positions,
        'end_positions': end_positions,
        'example_id': example_id,
        'offset_mapping': offset_mapping,
    }

def get_dataloaders(args):
    _ensure_data_root(args.data)
    name = 'squad_v2' if args.squad_version == 'v2' else 'squad'
    if args.rank == 0:
        print(f"[Data] Loading '{name}' under {args.data} (first run downloads; then reuses)")
    dl_mode = DownloadMode.FORCE_REDOWNLOAD if args.force_download else DownloadMode.REUSE_DATASET_IF_EXISTS
    raw = load_dataset(name, cache_dir=os.environ.get('HF_DATASETS_CACHE', args.data), download_mode=dl_mode)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True,
        cache_dir=os.environ.get('TRANSFORMERS_CACHE', args.data),
        force_download=args.force_download
    )

    train_features, val_features = _prepare_features(args, raw, tokenizer)

    # map id -> (context, gold answers)
    args._id2ex = {ex['id']:{'context':ex['context'], 'answers':ex['answers']} for ex in raw['validation']}
    args._tokenizer = tokenizer

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_features, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=args.drop_last_train)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_features, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=args.drop_last_val)

    train_loader = DataLoader(
        train_features, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=args.prefetch_factor,
        collate_fn=default_data_collator)
    val_loader = DataLoader(
        val_features, batch_size=args.batch_size, sampler=val_sampler, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=args.prefetch_factor,
        collate_fn=_collate_val)

    return train_loader, val_loader

# ------------------------- Metrics ------------------------------

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.count = 0.0
        self.avg = 0.0
        self.min = math.inf
        self.max = 0.0
    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1.0, self.count)
        self.min = min(self.min, val)
        self.max = max(self.max, val)
    def all_reduce(self):
        if dist.is_available() and dist.is_initialized():
            backend = dist.get_backend()
            device = torch.device(f"cuda:{torch.cuda.current_device()}") if backend == dist.Backend.NCCL else torch.device("cpu")
            t = torch.tensor([self.sum, self.count], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            self.sum, self.count = t.cpu().tolist()
            self.avg = self.sum / max(1.0, self.count)

# SQuAD EM/F1 helpers
import string, collections

def _norm(s: str) -> str:
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    # remove articles without regex (to keep this editor happy)
    words = [w for w in s.split() if w not in {"a","an","the"}]
    s = ' '.join(words)
    s = ' '.join(s.split())
    return s

def _em(pred: str, golds: List[str]) -> float:
    return float(any(_norm(pred) == _norm(g) for g in golds))

def _f1(pred: str, golds: List[str]) -> float:
    def score(a, b):
        at, bt = _norm(a).split(), _norm(b).split()
        common = collections.Counter(at) & collections.Counter(bt)
        ns = sum(common.values())
        if len(at) == 0 and len(bt) == 0: return 1.0
        if ns == 0: return 0.0
        p, r = ns/len(at), ns/len(bt)
        return 2*p*r/(p+r+1e-12)
    return max(score(pred, g) for g in golds) if len(golds) > 0 else score(pred, "")

# ------------------------- Train / Eval -------------------------

@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    losses = AverageMeter()
    em_sum = 0.0
    f1_sum = 0.0
    count  = 0

    tokenizer = args._tokenizer
    id2ex = args._id2ex

    def run_validation(dataloader):
        nonlocal em_sum, f1_sum, count
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            token_type_ids = batch['token_type_ids'].to(device, non_blocking=True)
            start_positions= batch['start_positions'].to(device, non_blocking=True)
            end_positions  = batch['end_positions'].to(device, non_blocking=True)

            if args.amp and device.type == 'cuda':
                with autocast():
                    out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                start_positions=start_positions, end_positions=end_positions, return_dict=True)
                    loss = out.loss
            else:
                out = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            start_positions=start_positions, end_positions=end_positions, return_dict=True)
                loss = out.loss
            losses.update(loss.item(), input_ids.size(0))

            s_logits = out.start_logits.cpu()
            e_logits = out.end_logits.cpu()

            # pick best span with constraints
            for i in range(input_ids.size(0)):
                offsets = batch['offset_mapping'][i]
                eid = batch['example_id'][i]
                golds = id2ex[eid]['answers']['text'] if len(id2ex[eid]['answers']['text']) > 0 else [""]
                context = id2ex[eid]['context']

                s = s_logits[i]; e = e_logits[i]
                max_len = args.max_answer_length
                best_score = -1e9; best_text = ""
                kS = min(args.n_best_size, s.numel())
                kE = min(args.n_best_size, e.numel())
                s_idx = torch.topk(s, k=kS).indices.tolist()
                e_idx = torch.topk(e, k=kE).indices.tolist()
                for si in s_idx:
                    for ej in e_idx:
                        if ej < si or (ej - si + 1) > max_len: continue
                        if offsets[si] == (0, 0) or offsets[ej] == (0, 0): continue
                        score = s[si].item() + e[ej].item()
                        if score > best_score:
                            best_score = score
                            st, en = offsets[si][0], offsets[ej][1]
                            best_text = context[st:en]
                if args.squad_version == 'v2':
                    cls_ids = (batch['input_ids'][i] == tokenizer.cls_token_id).nonzero(as_tuple=True)[0]
                    if cls_ids.numel() > 0:
                        idx = int(cls_ids[0].item())
                        null_score = s[idx].item() + e[idx].item()
                        if null_score - best_score > args.null_score_diff_threshold:
                            best_text = ""
                em_sum += _em(best_text, golds)
                f1_sum += _f1(best_text, golds)
                count  += 1

    run_validation(loader)

    # Reduce over ranks for the main shard
    if dist.is_available() and dist.is_initialized():
        backend = dist.get_backend()
        device0 = torch.device(f"cuda:{torch.cuda.current_device()}") if backend == dist.Backend.NCCL else torch.device("cpu")
        t = torch.tensor([losses.sum, losses.count, em_sum, f1_sum, float(count)], dtype=torch.float64, device=device0)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        loss_avg = (t[0] / max(1.0, t[1].item())).item()
        em = (t[2] / max(1.0, t[4].item())).item() * 100.0
        f1 = (t[3] / max(1.0, t[4].item())).item() * 100.0
    else:
        loss_avg = losses.avg
        em = (em_sum / max(1, count)) * 100.0
        f1 = (f1_sum / max(1, count)) * 100.0

    # Handle leftover items not covered by sampler (rare)
    if len(loader.sampler) * args.world_size < len(loader.dataset):
        aux_val_dataset = Subset(loader.dataset, range(len(loader.sampler) * args.world_size, len(loader.dataset)))
        aux_val_loader = DataLoader(aux_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=_collate_val)
        run_validation(aux_val_loader)
        # No distributed reduce here
        em = (em_sum / max(1, count)) * 100.0
        f1 = (f1_sum / max(1, count)) * 100.0

    return {'loss': float(loss_avg), 'em': float(em), 'f1': float(f1)}

def train_one_epoch(model, dataloader, optimizer, scheduler, device, scaler, args, epoch):
    model.train()
    losses = AverageMeter()
    step_time = AverageMeter()
    data_time = AverageMeter()

    if device.type == 'cuda':
        epoch_start = torch.cuda.Event(enable_timing=True)
        epoch_end   = torch.cuda.Event(enable_timing=True)
        epoch_start.record()
    else:
        epoch_start = time.perf_counter()

    step_start = time.perf_counter()
    samples_seen = 0.0
    total_steps = len(dataloader)

    for step_idx, batch in enumerate(dataloader):
        cur = time.perf_counter()
        data_time.update(cur - step_start, n=1)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        bs = batch['input_ids'].size(0)
        samples_seen += bs

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                out = model(**batch)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()

        losses.update(loss.item(), bs)
        elapsed = time.perf_counter() - step_start
        step_time.update(elapsed, n=1)
        step_start = time.perf_counter()

        # Per-step progress logging
        if args.rank == 0 and (step_idx == 0 or (step_idx + 1) % args.log_interval == 0 or (step_idx + 1) == total_steps):
            inst_tp = bs / max(1e-9, elapsed)
            print(
                f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}][Epoch {epoch:03d} Step {step_idx+1:05d}/{total_steps}] "
                f"loss={loss.item():.4f} avg_loss={losses.avg:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f} "
                f"step_time={elapsed:.3f}s data={data_time.avg:.3f}s comp={max(0.0, step_time.avg - data_time.avg):.3f}s "
                f"ips=~{inst_tp:.1f}",
                flush=True
            )

    if device.type == 'cuda':
        epoch_end.record(); epoch_end.synchronize()
        duration = epoch_start.elapsed_time(epoch_end) / 1000.0
    else:
        duration = time.perf_counter() - epoch_start

    throughput = samples_seen / max(1e-6, duration)

    local_loss = losses.avg
    losses.all_reduce()

    return {
        'loss_global' : float(losses.avg),
        'loss': float(local_loss),
        'step_time_min': float(step_time.min),
        'step_time_max': float(step_time.max),
        'step_time': float(step_time.avg),
        'data_time': float(data_time.avg),
        'comp_time': float(step_time.avg - data_time.avg),
        'epoch_time': float(duration),
        'throughput': float(throughput),
    }

def save_log(path, log):
    """Atomically write log dict to JSON file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def train(args):
    device = torch.device(args.device)

    # Data
    train_loader, val_loader = get_dataloaders(args)

    # Model
    config = AutoConfig.from_pretrained(
        args.model_name,
        cache_dir=os.environ.get('TRANSFORMERS_CACHE', args.data),
        force_download=args.force_download
    )
    model  = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name, config=config,
        cache_dir=os.environ.get('TRANSFORMERS_CACHE', args.data),
        force_download=args.force_download,
    ).to(device)

    model = DDP(
        model,
        device_ids=[args.local_rank] if device.type == "cuda" else None,
        gradient_as_bucket_view=True,
        find_unused_parameters=False,
        static_graph=args.static_graph
    )

    print(f"Model '{args.model_name}' initialized.", flush=True)

    # Straggle sim
    straggle_sim = SlowWorkerPattern(points=args.straggle_points, prob=args.straggle_prob, amount=args.straggle_amount,
                                    ranks=args.straggle_ranks, multiplier_range=args.straggle_multiply, seed=42,
                                    verbose=args.straggle_verbose)
    if straggle_sim.attach(model): print(f"Straggle sim initialized with {straggle_sim}")
    else: print(f"Straggle sim inactive")

    # Optim & sched
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, foreach=True)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and device.type == "cuda"))

    def now(): return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    best_em = 0.0
    best_f1 = 0.0

    cfg = {k: v for k, v in vars(args).items() if not k.startswith('_')}
    log = {"time": now(), "config": cfg, "epochs": {}}
    save_log(args.json, log)

    for epoch in range(args.epochs):
        print(f"[{now()}][Epoch {epoch:03d}] ...")
        epoch_start = time.time()

        straggle_sim.reset_stats()

        train_loader.sampler.set_epoch(epoch)

        # Train for one epoch and get metrics
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler, args, epoch)

        # Validate and get metrics
        val_metrics  = validate(model, val_loader, device, args)

        # Calculate total epoch time
        epoch_time = time.time() - epoch_start

        # Current LR (after per-step schedule)
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"[{now()}][Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} (global={train_metrics['loss_global']:.4f}) "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_em={val_metrics['em']:.2f}% val_f1={val_metrics['f1']:.2f}% "
            f"lr={current_lr:.6f} epoch_time={epoch_time:.2f}s step_time={train_metrics['step_time']:.2f}s "
            f"(min={train_metrics['step_time_min']:.2f}s, max={train_metrics['step_time_max']:.2f}s) "
            f"tp=~{train_metrics['throughput']:.1f} samples/s "
            f"straggle_events={straggle_sim.get_stats().get('num_straggle_events', 0)}",
            flush=True
        )

        # Log JSON (match DenseNet shape + QA fields)
        epoch_metrics = {
            # Training metrics
            "lr": float(current_lr),
            "train_loss": float(train_metrics['loss']),
            "train_loss_global": float(train_metrics['loss_global']),
            "steps": int(len(train_loader)),
            "step_time_min": float(train_metrics['step_time_min']),
            "step_time_max": float(train_metrics['step_time_max']),
            "step_time": float(train_metrics['step_time']),
            "data_time": float(train_metrics['data_time']),
            "comp_time": float(train_metrics['comp_time']),
            "epoch_time": float(epoch_time),
            "epoch_train_time": float(train_metrics['epoch_time']),
            "epoch_train_throughput": float(train_metrics['throughput']),

            # Validation metrics
            "val_loss": float(val_metrics['loss']),
            "val_em": float(val_metrics['em']),
            "val_f1": float(val_metrics['f1']),

            # straggle-sim
            "straggle": straggle_sim.get_stats() if straggle_sim.active else {}
        }

        log["epochs"][str(epoch)] = epoch_metrics
        save_log(args.json, log)

        best_em = max(best_em, val_metrics['em'])
        best_f1 = max(best_f1, val_metrics['f1'])

        # NOTE: scheduler stepped per-batch in train_one_epoch (do not step here)

# ------------------------- Entry / Setup ------------------------

def setup_ddp(args):
    # Ensure args contains everything we need. Give priority to ENV vars
    def env_int(key, default): return default if os.environ.get(key) in (None, "") else int(os.environ.get(key))
    def env_str(key, default): return default if os.environ.get(key) in (None, "") else os.environ.get(key)

    args.rank        = env_int("RANK", args.rank)
    args.world_size  = env_int("WORLD_SIZE", args.world_size)
    args.master_addr = env_str("MASTER_ADDR", args.master_addr)
    args.master_port = env_int("MASTER_PORT", args.master_port)
    args.iface       = env_str("IFACE", args.iface)

    # Respect LOCAL_RANK if present (torchrun)
    env_local_rank = os.environ.get("LOCAL_RANK")
    if env_local_rank is not None:
        args.local_rank = int(env_local_rank)
    elif torch.cuda.device_count():
        args.local_rank = (args.rank % torch.cuda.device_count())
    else:
        args.local_rank = 0

    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)

    # Ensure the variables torch.distributed expects are present.
    os.environ.setdefault("RANK",        str(args.rank))
    os.environ.setdefault("WORLD_SIZE",  str(args.world_size))
    os.environ.setdefault("MASTER_ADDR", args.master_addr)
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ.setdefault("LOCAL_RANK",  str(args.local_rank))

    os.environ.setdefault("GLOO_SOCKET_IFNAME", args.iface)
    os.environ.setdefault("GLOO_SOCKET_NTHREADS", "8")
    os.environ.setdefault("GLOO_NSOCKS_PERTHREAD", "2")
    os.environ.setdefault("GLOO_BUFFSIZE", "8388608")

    os.environ.setdefault("NCCL_SOCKET_IFNAME", args.iface)               # e.g. ens4f0
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_DEBUG_SUBSYS", "INIT,NET,ENV")
    os.environ.setdefault("NCCL_DEBUG_FILE", f"/tmp/nccl_%h_rank{os.environ.get('RANK','0')}.log")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")         # P100 P2P is limited
    os.environ.setdefault("NCCL_TREE_THRESHOLD", "0")      # Force ring for stability
    os.environ.setdefault("NCCL_IB_DISABLE", "0")          # Enable IB if available on 100G
    os.environ.setdefault("NCCL_BUFFSIZE", "8388608")
    os.environ.setdefault("NCCL_SOCKET_NTHREADS", "4")  # More NCCL threads
    os.environ.setdefault("NCCL_NSOCKS_PERTHREAD", "4")

    # Start the process group
    dist.init_process_group(backend=args.backend, init_method="env://", rank=args.rank, world_size=args.world_size, timeout=datetime.timedelta(seconds=30))

    if args.rank == 0:
        print(f"[DDP] backend={args.backend} world_size={args.world_size} "
              f"master={args.master_addr}:{args.master_port} iface={args.iface} local_rank={args.local_rank}", flush=True)

def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()

    # DDP/System
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--iface', type=str, default="ens4f0")
    parser.add_argument('--master_addr', type=str, default="42.0.0.1")
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument("--backend", type=str, default="gloo", help="DDP backend (e.g., gloo, nccl)")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument("--deterministic", action='store_true')
    parser.add_argument("--workers", type=int, default=4)

    # Training/model
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help="HF model for QA")
    parser.add_argument('--data', type=str, required=True, help='Single directory (created if missing). Everything stays under it.')
    parser.add_argument('--force_download', action='store_true', help='Force re-download of the SQuAD dataset into the --data cache')

    parser.add_argument('--squad_version', type=str, choices=['v1','v2'], default='v1')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    parser.add_argument("--drop_last_train", action='store_true', help="Drop last from train dataset")
    parser.add_argument("--drop_last_val", action='store_true', help="Drop last from val dataset")
    parser.add_argument("--static_graph", action='store_true', help="Enable static_graph in DDP")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument('--log_interval', type=int, default=50, help="Steps between progress prints")

    # QA tokenization / decode
    parser.add_argument('--max_seq_len', type=int, default=384)
    parser.add_argument('--doc_stride', type=int, default=128)
    parser.add_argument('--max_answer_length', type=int, default=30)
    parser.add_argument('--n_best_size', type=int, default=20)
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0)

    # Straggle
    def csv_ints(s: str) -> List[int]:
        if not s: return []
        try: return [int(x) for x in re.split(r"\s*,\s*", s) if x]
        except ValueError: raise argparse.ArgumentTypeError("Expected a comma-separated list of integers (e.g. 1,2,3)")
    parser.add_argument("--straggle_points", type=int, help="Number of straggle points (1-3). Use 0 for no straggle sim", default=0)
    parser.add_argument("--straggle_prob", type=float, help="Probability to straggle at each point", default=0)
    parser.add_argument("--straggle_ranks", type=csv_ints, help="comma separated list of ints", default=[])
    parser.add_argument("--straggle_amount", type=float, help="base straggle amount in seconds (e.g. mean step time)", default=0)
    parser.add_argument("--straggle_multiply", type=float, nargs=2, metavar=("lo","hi"), help="straggle amount multipler lo and hi", default=[1.0, 1.0])
    parser.add_argument("--straggle_verbose", action='store_true')

    # Logging
    parser.add_argument("--json", type=str, default="bert_squad.json", help="Path to JSON run log")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # Args sanity checks/corrections
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        if args.rank == 0: print("[Info] Using device=cpu because CUDA is not available", flush=True)
    if args.amp and args.device == 'cpu':
        args.amp = False
        if args.rank == 0: print("[Info] Disabling AMP because CUDA is not available", flush=True)
    if args.workers < 1:
        if args.rank == 0: print("[Info] Workers requested < 1; using workers=1", flush=True)
        args.workers = 1

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    setup_ddp(args)

    if args.deterministic:
        args.seed = 42 + args.rank
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    cfg = {k: v for k, v in vars(args).items() if not k.startswith('_')}
    print(json.dumps(cfg, indent=2))
    try:
        train(args)
    finally:
        cleanup()

if __name__ == '__main__':
    main()
