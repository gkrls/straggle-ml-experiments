#!/usr/bin/env python3
import argparse, os, sys, json, time, datetime, random, string
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset

# ------------------------- Tokenizer ------------------------------
def simple_tokenizer(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

PAD_ID = 0
UNK_ID = 1

# ------------------------- HF dataset helpers ---------------------
def hf_load_train_val(data_path, val_fraction=0.01, seed=42, cache_dir=None):
    """
    Load your local OpenWebText dataset via HF Datasets.
      - data_path: directory that contains the loader .py (e.g., DATA_ROOT) OR the loader .py file itself
      - the loader itself finds shards under DATA_ROOT/openwebtext/*.xz
      - cache_dir controls where HF stores Arrow cache
    """
    ds = load_dataset(path=str(data_path), name="plain_text", split="train",
                      cache_dir=cache_dir, trust_remote_code=True)
    splits = ds.train_test_split(test_size=val_fraction, seed=seed, shuffle=True)
    return splits["train"], splits["test"]

def build_vocab_from_hfds(ds_train, max_vocab=50000, bytes_limit=2_000_000_000):
    counter = Counter()
    seen = 0
    for ex in ds_train:
        txt = ex["text"]
        counter.update(simple_tokenizer(txt))
        seen += len(txt)
        if bytes_limit and seen >= bytes_limit:
            break
    vocab = {"<pad>": PAD_ID, "<unk>": UNK_ID}
    for w, _ in counter.most_common(max_vocab - len(vocab)):
        vocab[w] = len(vocab)
    return vocab

# ------------------------- IterableDataset ------------------------
class HFWindows(IterableDataset):
    """
    Streams (x, y) windows (seq_len+1) from HF Dataset of documents.
    DDP- and multi-worker-aware via modulo partitioning.
    """
    def __init__(self, data_path, split, vocab, seq_len=256, stride=128,
                 world_size=1, rank=0, seed=42, val_fraction=0.01, cache_dir=None):
        super().__init__()
        self.data_path = data_path
        self.split = split    # "train" or "val"
        self.vocab = vocab
        self.seq_len = seq_len
        self.stride = stride
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.val_fraction = val_fraction
        self.cache_dir = cache_dir

    def _encode_line(self, line):
        return [self.vocab.get(t, UNK_ID) for t in simple_tokenizer(line)]

    def _yield_windows(self, toks):
        T = self.seq_len + 1
        for i in range(0, max(0, len(toks) - T + 1), self.stride):
            w = toks[i:i+T]
            if len(w) == T:
                x = torch.tensor(w[:-1], dtype=torch.long)
                y = torch.tensor(w[1:],  dtype=torch.long)
                yield x, y

    def _iter_doc(self, text):
        buf = self._encode_line(text)
        for xy in self._yield_windows(buf):
            yield xy

    def __iter__(self):
        # Load per worker/process so HF Dataset objects aren’t pickled
        ds_train, ds_val = hf_load_train_val(self.data_path, self.val_fraction, self.seed, cache_dir=self.cache_dir)
        ds = ds_train if self.split == "train" else ds_val

        info = torch.utils.data.get_worker_info()
        num_workers = info.num_workers if info else 1
        worker_id = info.id if info else 0
        consumers = self.world_size * num_workers
        consumer_id = self.rank * num_workers + worker_id

        i = 0
        for ex in ds:
            if (i % consumers) == consumer_id:
                for xy in self._iter_doc(ex["text"]):
                    yield xy
            i += 1

# ------------------------- Model ---------------------------------
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2,
                 dropout_prob=0.5, tie_weights=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_prob if num_layers > 1 else 0.0)
        if tie_weights and embed_dim == hidden_dim:
            self.decoder = nn.Linear(embed_dim, vocab_size, bias=False)
            self.decoder.weight = self.embedding.weight
        else:
            self.decoder = nn.Linear(hidden_dim, vocab_size)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        if getattr(self.decoder, "bias", None) is not None:
            nn.init.zeros_(self.decoder.bias)

    def forward(self, x, hidden=None):
        h = self.dropout(self.embedding(x))
        out, hidden = self.lstm(h, hidden)
        out = self.dropout(out)
        return self.decoder(out), hidden

# ------------------------- Metrics helper ------------------------
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

# ------------------------- Train / Eval --------------------------
@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    losses = AverageMeter()
    ce = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction='sum').to(device)

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if args.amp and device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                logits, _ = model(inputs)
                loss_sum = ce(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        else:
            logits, _ = model(inputs)
            loss_sum = ce(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        ntok = targets.numel()
        losses.update(loss_sum.item(), ntok)

    losses.all_reduce()
    val_loss = losses.avg  # per-token NLL
    val_ppl = float(np.exp(np.clip(val_loss, 0, 20)))
    return {'val_loss': val_loss, 'val_ppl': val_ppl}

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
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

    for inputs, targets in dataloader:
        data_time.update(time.perf_counter() - step_start, n=1)

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                logits, _ = model(inputs)
                loss_sum = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                ntok = targets.numel()
                loss = loss_sum / ntok
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer); scaler.update()
        else:
            logits, _ = model(inputs)
            loss_sum = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            ntok = targets.numel()
            (loss_sum / ntok).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        losses.update(loss_sum.item(), ntok)
        tokens_seen += int(ntok)
        batches += 1

        step_time.update(time.perf_counter() - step_start, n=1)
        step_start = time.perf_counter()

    if device.type == 'cuda':
        epoch_end.record(); epoch_end.synchronize()
        duration = epoch_start.elapsed_time(epoch_end) / 1000.0
    else:
        duration = time.perf_counter() - epoch_start

    tok_per_sec = tokens_seen / max(1e-6, duration)

    local_loss = losses.avg
    losses.all_reduce()
    train_loss_global = losses.avg
    train_ppl = float(np.exp(np.clip(train_loss_global, 0, 20)))

    return {
        'train_loss_global': train_loss_global,   # per-token
        'train_loss': local_loss,                 # per-token
        'train_step_time': step_time.avg,
        'train_data_time': data_time.avg,
        'train_comp_time': step_time.avg - data_time.avg,
        'epoch_duration': duration,
        'epoch_throughput': tok_per_sec,         # tokens/sec
        'tokens_seen': tokens_seen,
        'train_ppl': train_ppl,
        'batches': batches,
    }

def save_log(path, log):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def _resolve_repo_root_and_cache_dir(data_path, cache_dir_arg):
    """
    data_path can be:
      - a directory (preferred): DATA_ROOT (contains openwebtext.py and openwebtext/*.xz)
      - a file: path/to/openwebtext.py
    Default cache dir becomes <DATA_ROOT>/cache
    """
    p = Path(data_path).resolve()
    # dataset_root = directory that holds openwebtext.py
    dataset_root = p.parent if p.is_file() else p

    shards_dir = dataset_root / "openwebtext"   # where *.xz live

    if cache_dir_arg is None:
        cache_dir = dataset_root / "cache"      # <-- CHANGED: under root now
    else:
        cache_dir = Path(cache_dir_arg)

    cache_dir.mkdir(parents=True, exist_ok=True)
    return dataset_root, shards_dir, cache_dir

def train(args):
    device = torch.device(args.device)

    # Resolve paths + cache
    repo_root, shards_dir, cache_dir = _resolve_repo_root_and_cache_dir(args.data, args.cache_dir)
    args.cache_dir = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = args.cache_dir

    if args.rank == 0:
        print(f"Loading HF dataset from {args.data} (cache: {args.cache_dir}) ...", flush=True)

    # Load/train/val + build vocab from TRAIN
    ds_train, ds_val = hf_load_train_val(args.data, args.val_fraction, seed=42, cache_dir=args.cache_dir)
    vocab = build_vocab_from_hfds(ds_train, max_vocab=args.max_vocab, bytes_limit=args.vocab_bytes)

    # Iterable datasets / loaders (DDP-aware)
    train_ds = HFWindows(args.data, "train", vocab, seq_len=args.seq_len, stride=args.stride,
                         world_size=args.world_size, rank=args.rank, seed=42,
                         val_fraction=args.val_fraction, cache_dir=args.cache_dir)
    val_ds   = HFWindows(args.data, "val",   vocab, seq_len=args.seq_len, stride=args.seq_len,
                         world_size=args.world_size, rank=args.rank, seed=42,
                         val_fraction=args.val_fraction, cache_dir=args.cache_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.workers,
                              pin_memory=True, persistent_workers=(args.workers>0))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, num_workers=args.workers,
                              pin_memory=True, persistent_workers=(args.workers>0))

    # Model
    model = LSTMLanguageModel(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout_prob=args.dropout,
        tie_weights=args.tie_weights
    ).to(device)

    model = DDP(model, device_ids=[args.local_rank] if device.type == "cuda" else None,
                gradient_as_bucket_view=True, find_unused_parameters=False, static_graph=args.static_graph)

    if args.rank == 0:
        print(f"\nDATA (loader path): {args.data}")
        print(f"Shards dir: {shards_dir}")
        print(f"Cache dir:  {args.cache_dir}")
        print(f"  Vocab size: {len(vocab):,}")
        print(f"  Seq len: {args.seq_len} | Stride: {args.stride}")
        print(f"\nModel: LSTM LM ({args.num_layers} layers, tie={args.tie_weights})")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("=" * 60, flush=True)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction='sum').to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, foreach=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp) if device.type == "cuda" else None

    def now(): return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    best_ppl = float('inf')

    if args.rank == 0:
        if args.json is None: args.json = "lstm_lm_openwebtext_hf.json"
        log = {"time": now(), "data": str(args.data), "cache_dir": args.cache_dir,
               "config": vars(args), "vocab_size": len(vocab), "epochs": {}}
        save_log(args.json, log)

    for epoch in range(args.epochs):
        if args.rank == 0: print(f"[{now()}][Epoch {epoch:03d}] ...", flush=True)
        epoch_start = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics   = validate(model, val_loader, device, args)

        epoch_time = time.time() - epoch_start
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
                "train_throughput": float(train_metrics['epoch_throughput']),   # tokens/s
                "val_loss": float(val_metrics['val_loss']),
                "val_ppl": float(val_metrics['val_ppl']),
                "lr": float(current_lr),
                "epoch_time": float(epoch_time),
                "epoch_throughput": float(train_metrics['epoch_throughput']),
                "steps": int(train_metrics['batches'])
            }
            with open(args.json, "r") as f: log = json.load(f)
            log["epochs"][str(epoch)] = epoch_metrics
            save_log(args.json, log)

            if val_metrics['val_ppl'] < best_ppl:
                best_ppl = val_metrics['val_ppl']

        scheduler.step()

    if args.rank == 0:
        print(f"\n[{now()}] Training completed!")
        print(f"Best validation perplexity: {best_ppl:.2f}")

# ------------------------- DDP setup -----------------------------
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
    dist.init_process_group(backend=args.backend, init_method="env://", rank=args.rank,
                            world_size=args.world_size, timeout=datetime.timedelta(seconds=30))
    if args.rank == 0:
        print(f"[DDP] backend={args.backend} world_size={args.world_size} "
              f"master={args.master_addr}:{args.master_port} iface={args.iface} local_rank={args.local_rank}", flush=True)

def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

# ------------------------- Entry --------------------------------
def main():
    p = argparse.ArgumentParser("Distributed LSTM LM on HF OpenWebText (local Zenodo shards)")
    # DDP/system
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world_size', type=int, default=1)
    p.add_argument('--iface', type=str, default="ens4f0")
    p.add_argument('--master_addr', type=str, default="42.0.0.1")
    p.add_argument("--master_port", type=int, default=29500)
    p.add_argument("--backend", type=str, default="gloo", choices=['gloo', 'nccl'])
    p.add_argument("--device", type=str, choices=['cuda','cpu'], default='cuda')
    p.add_argument("--deterministic", action='store_true')
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--json", type=str, default=None)
    # Data (HF)
    p.add_argument('--data', type=str, required=True, help='Directory (DATA_ROOT) or loader .py path')
    p.add_argument('--cache_dir', type=str, default=None, help='HF datasets cache dir (default: <dataset_root>/cache)')
    p.add_argument('--max_vocab', type=int, default=50000)
    p.add_argument('--vocab_bytes', type=int, default=2_000_000_000)
    p.add_argument('--seq_len', type=int, default=256)
    p.add_argument('--stride', type=int, default=128)
    p.add_argument('--val_fraction', type=float, default=0.01)
    # Training
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--step_size', type=int, default=5)
    p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--static_graph", action='store_true')
    p.add_argument("--prefetch_factor", type=int, default=2)
    # Model
    p.add_argument('--embed_dim', type=int, default=512)
    p.add_argument('--hidden_dim', type=int, default=512)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--tie_weights', action='store_true')

    args = p.parse_args()

    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        print("[Info] Using device=cpu because CUDA is not available", flush=True)
    if args.amp and args.device == 'cpu':
        args.amp = False
        print("[Info] Disabling AMP because CUDA is not available", flush=True)
    if args.workers < 1:
        print("[Info] Workers requested < 1; using workers=1", flush=True)
        args.workers = 1

    sys.stdout.reconfigure(line_buffering=True)
    setup_ddp(args)
    try:
        train(args)
    finally:
        cleanup()

if __name__ == "__main__":
    main()
