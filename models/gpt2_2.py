#!/usr/bin/env python3
import argparse, os, sys, json, time, datetime, random, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

# ------------------------- Simple Parquet resolver -------------------------
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

def hf_load_train_val_parquet(data_root: Path, val_fraction=0.01, seed=42, cache_dir=None):
    train_glob, val_glob, split_needed = _resolve_parquet_layout(data_root)
    if split_needed:
        ds = load_dataset("parquet", data_files={"train": train_glob}, cache_dir=cache_dir)
        splits = ds["train"].train_test_split(test_size=val_fraction, seed=seed, shuffle=True)
        return splits["train"], splits["test"]
    else:
        ds = load_dataset("parquet", data_files={"train": train_glob, "validation": val_glob}, cache_dir=cache_dir)
        return ds["train"], ds["validation"]

# ------------------------- Tokenize like your student ----------------------
def build_tokenized_splits(ds_train, tokenizer, block_size=512, val_fraction=0.01, seed=42):
    # split at doc level
    split = ds_train.train_test_split(test_size=val_fraction, seed=seed)
    tr, va = split["train"], split["test"]

    def tok_fn(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=block_size,
            return_attention_mask=True,
        )
        # labels = input_ids, but mask pads to -100 so loss ignores them
        out["labels"] = [ids[:] for ids in out["input_ids"]]
        for i in range(len(out["labels"])):
            am = out["attention_mask"][i]
            lbl = out["labels"][i]
            pad_mask = [1 if a == 0 else 0 for a in am]
            for j in range(len(lbl)):
                if pad_mask[j]:
                    lbl[j] = -100
        return out

    tr_tok = tr.map(tok_fn, batched=True, remove_columns=tr.column_names)
    va_tok = va.map(tok_fn, batched=True, remove_columns=va.column_names)
    return tr_tok, va_tok

# ------------------------- DDP-aware IterableDataset -----------------------
class ShardedIterable(IterableDataset):
    """
    Iterates a tokenized HF dataset deterministically, sharded by (rank, workers),
    and yields dicts with torch tensors: input_ids, attention_mask, labels.
    """
    def __init__(self, ds, world_size=1, rank=0, seed=1337):
        super().__init__()
        self.ds = ds
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _index_perm(self, n, rng: random.Random):
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
        for i in self._index_perm(n, rng):
            if (i % consumers) != consumer_id:
                continue
            ex = self.ds[i]
            yield {
                "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(ex["labels"], dtype=torch.long),
            }

# ------------------------- Meters -----------------------------------------
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

# ------------------------- Train / Val -------------------------------------
@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    losses = AverageMeter()
    for batch in loader:
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)  # padded labels are -100

        if args.amp and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss  # already mean over non-ignored labels
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
        ntok_eff = int((labels != -100).sum())
        losses.update(loss.item(), ntok_eff)

    losses.all_reduce()
    val_loss = losses.avg
    val_ppl  = float(np.exp(np.clip(val_loss, 0, 25)))
    return {'val_loss': val_loss, 'val_ppl': val_ppl}

def train_one_epoch(model, dataloader, optimizer, device, scaler, args):
    model.train()
    losses = AverageMeter()
    step_time = AverageMeter()
    data_time = AverageMeter()

    if device.type == 'cuda':
        epoch_start = torch.cuda.Event(enable_timing=True)
        epoch_end = torch.cuda.Event(enable_timing=True)
        epoch_start.record()
    else:
        epoch_start = time.perf_counter()

    step_start = time.perf_counter()
    tokens_seen = 0
    batches = 0

    for batch in dataloader:
        data_time.update(time.perf_counter() - step_start, n=1)

        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer); scaler.update()
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        ntok_eff = int((labels != -100).sum())
        losses.update(loss.item(), ntok_eff)
        tokens_seen += ntok_eff
        batches += 1

        step_time.update(time.perf_counter() - step_start, n=1)
        step_start = time.perf_counter()

        if args.steps_per_epoch and batches >= args.steps_per_epoch:
            break

    if device.type == 'cuda':
        epoch_end.record(); epoch_end.synchronize()
        duration = epoch_start.elapsed_time(epoch_end) / 1000.0
    else:
        duration = time.perf_counter() - epoch_start

    tok_per_sec = tokens_seen / max(1e-6, duration)

    local_loss = losses.avg
    losses.all_reduce()
    train_loss_global = losses.avg
    train_ppl = float(np.exp(np.clip(train_loss_global, 0, 25)))

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
    }

def save_log(path, log):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

# ------------------------- DDP setup ---------------------------------------
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
    dist.init_process_group(backend=args.backend, init_method="env://", rank=args.rank,
                            world_size=args.world_size, timeout=datetime.timedelta(seconds=60))
    if args.rank == 0:
        print(f"[DDP] backend={args.backend} world_size={args.world_size} "
              f"master={args.master_addr}:{args.master_port} iface={args.iface} local_rank={args.local_rank}", flush=True)

def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

# ------------------------- Train driver ------------------------------------
def train(args):
    device = torch.device(args.device)

    data_root = Path(args.data).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"--data path not found: {data_root}")

    cache_dir = Path(args.cache_dir) if args.cache_dir is not None else (data_root / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)

    # Tokenizer like student's setup
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    PAD_ID = tokenizer.pad_token_id
    vocab_size = tokenizer.vocab_size  # 50257

    if args.rank == 0:
        print(f"Loading Parquet dataset from {data_root} (cache: {cache_dir}) ...", flush=True)
    ds_train, ds_val = hf_load_train_val_parquet(data_root, val_fraction=args.val_fraction, seed=args.seed, cache_dir=str(cache_dir))
    # Tokenize with padding/truncation (student-style), but labels masked on pads
    tr_tok, va_tok = build_tokenized_splits(ds_train, tokenizer, block_size=args.seq_len, val_fraction=args.val_fraction, seed=args.seed)

    # Sharded iterable datasets
    train_ds = ShardedIterable(tr_tok, world_size=args.world_size, rank=args.rank, seed=args.seed)
    val_ds   = ShardedIterable(va_tok, world_size=args.world_size, rank=args.rank, seed=args.seed+1)

    def _seed_worker(worker_id):
        import numpy as _np, random as _random
        worker_seed = (args.seed + args.rank * max(1, args.workers) + worker_id) % 2**32
        _np.random.seed(worker_seed); _random.seed(worker_seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.workers,
                              pin_memory=True, persistent_workers=(args.workers>0),
                              worker_init_fn=_seed_worker)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, num_workers=args.workers,
                              pin_memory=True, persistent_workers=(args.workers>0),
                              worker_init_fn=_seed_worker)

    # Model like student's config
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=args.seq_len, n_ctx=args.seq_len,
        n_embd=384, n_layer=6, n_head=6, n_inner=1536,
        activation_function="gelu",
        resid_pdrop=args.dropout, embd_pdrop=args.dropout, attn_pdrop=args.dropout,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=PAD_ID,
        use_cache=False,
    )
    model = GPT2LMHeadModel(cfg)
    # ensure embedding matches tokenizer (pad token exists)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    model = DDP(model, device_ids=[args.local_rank] if device.type == "cuda" else None,
                gradient_as_bucket_view=True, find_unused_parameters=False, static_graph=args.static_graph)

    if args.rank == 0:
        n_tr = len(tr_tok); n_va = len(va_tok)
        print(f"Model 'GPT2LMHeadModel' initialized.")
        print(f"\nDATA ROOT: {data_root}")
        print(f"  Train docs: {len(ds_train):,} | Val docs: {len(ds_val):,}")
        print(f"  Tokenized train samples: {n_tr:,} | val samples: {n_va:,}")
        print(f"  Vocab size: {len(tokenizer):,}")
        print(f"  Seq len: {args.seq_len} | (padded/truncated)")
        print(f"\nModel: GPT-2 from scratch (6 layers, 6 heads, d_model=384)")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("=" * 60, flush=True)

    # Optimizer & scheduler like student
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, args.steps_per_epoch) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp) if device.type == "cuda" else None

    def now(): return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    best_ppl = float('inf')

    if args.rank == 0:
        if args.json is None: args.json = "gpt2_student_style.json"
        log = {"time": now(), "data_root": str(data_root), "cache_dir": str(cache_dir),
               "config": vars(args), "vocab_size": len(tokenizer), "epochs": {}}
        save_log(args.json, log)

    for epoch in range(args.epochs):
        if args.rank == 0: print(f"[{now()}][Epoch {epoch:03d}] ...", flush=True)

        train_ds.set_epoch(epoch)
        val_ds.set_epoch(epoch)  # fine either way; val order doesn’t matter

        # train
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, scaler, args)
        # step-based scheduler progressed inside train loop? we used per-step in student style
        # here we call step() once per batch inside the loop would require passing scheduler; simpler: call here for the whole epoch steps:
        # but we used total_steps; better: advance per batch in train loop – add scheduler.step() there:
        # To match student closely, add per-batch step:
        # (we’ll emulate by calling step() train_metrics['batches'] times)
        for _ in range(train_metrics['batches']):
            scheduler.step()

        # validate
        val_metrics   = validate(model, val_loader, device, args)

        epoch_time = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        if args.rank == 0:
            print(f"[{now()}][Epoch {epoch:03d}] "
                  f"train_loss/tok={train_metrics['train_loss']:.4f} "
                  f"train_ppl={train_metrics['train_ppl']:.2f} "
                  f"val_loss/tok={val_metrics['val_loss']:.4f} "
                  f"val_ppl={val_metrics['val_ppl']:.2f} "
                  f"lr={current_lr:.6f} time={epoch_time:.2f}s tok/s~{train_metrics['epoch_throughput']:.0f}",
                  flush=True)

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
            with open(args.json, "r") as f: log = json.load(f)
            log["epochs"][str(epoch)] = epoch_metrics
            save_log(args.json, log)

            if val_metrics['val_ppl'] < best_ppl:
                best_ppl = val_metrics['val_ppl']

    if args.rank == 0:
        print(f"\n[{now()}] Training completed!")
        print(f"Best validation perplexity: {best_ppl:.2f}")

# ------------------------- Entry ------------------------------------------
def main():
    p = argparse.ArgumentParser("GPT-2 on OpenWebText — student-style (padded blocks)")
    # DDP/system
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world_size', type=int, default=1)
    p.add_argument('--iface', type=str, default="ens4f0")
    p.add_argument('--master_addr', type=str, default="42.0.0.1")
    p.add_argument("--master_port", type=int, default=29500)
    p.add_argument("--backend", type=str, default="gloo", choices=['gloo', 'nccl'])
    p.add_argument("--device", type=str, choices=['cuda','cpu'], default='cuda')
    p.add_argument("--deterministic", action='store_true')
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--json", type=str, default=None)
    p.add_argument('--seed', type=int, default=1337)
    # Data
    p.add_argument('--data', type=str, required=True)
    p.add_argument('--cache_dir', type=str, default=None)
    p.add_argument('--val_fraction', type=float, default=0.01)
    # Training
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--steps_per_epoch', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=32)  # PER-GPU, not divided
    p.add_argument('--learning_rate', type=float, default=1e-4)  # student used 1e-4; safer than 3e-4
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--static_graph", action='store_true')
    p.add_argument("--prefetch_factor", type=int, default=2)
    # Model/window
    p.add_argument('--seq_len', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.1)
    args = p.parse_args()

    # (optional) determinism settings
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    else:
        torch.backends.cudnn.benchmark = True

    # TF32 on Ampere+ (P100 ignores)
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'; print("[Info] Using device=cpu because CUDA is not available", flush=True)
    if args.amp and args.device == 'cpu':
        args.amp = False; print("[Info] Disabling AMP because CUDA is not available", flush=True)
    if args.workers < 1:
        args.workers = 1

    sys.stdout.reconfigure(line_buffering=True)
    setup_ddp(args)
    print(json.dumps(vars(args), indent=2))
    try:
        train(args)
    finally:
        cleanup()

if __name__ == "__main__":
    main()
