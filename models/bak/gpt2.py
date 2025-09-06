#!/usr/bin/env python3
"""
GPT-2 (124M) DDP trainer on local Parquet OpenWebText.
"""

import os, sys, argparse, time, datetime, json, math, random, warnings, logging
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW

from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from datasets import load_dataset


# ------------------------- small helpers -------------------------
def now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def save_log(path, log):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = 0.0; self.count = 0.0; self.avg = 0.0
    def update(self, val, n=1): self.sum += float(val)*n; self.count += n; self.avg = self.sum / max(1.0, self.count)
    def all_reduce(self):
        if dist.is_available() and dist.is_initialized():
            backend = dist.get_backend()
            device = torch.device(f"cuda:{torch.cuda.current_device()}") if backend == dist.Backend.NCCL else torch.device("cpu")
            t = torch.tensor([self.sum, self.count], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            self.sum, self.count = t.cpu().tolist()
            self.avg = self.sum / max(1.0, self.count)


# ------------------------- parquet helpers -------------------------
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


# ------------------------- DDP-aware dataset -------------------------
class GPT2Windows(IterableDataset):
    """Streams (seq_len+1) windows packed across docs with EOS separators."""
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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Token indices sequence length is longer than.*")
            logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
            ids = self.tok.encode(line, add_special_tokens=False)
            logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.WARNING)
        if self.append_eos and (not ids or ids[-1] != self.EOS_ID):
            ids.append(self.EOS_ID)
        return ids

    def _index_permutation(self, n, rng: random.Random):
        if n <= 1:
            for j in range(n): yield j
            return
        start = rng.randrange(n)
        while True:
            step = rng.randrange(1, n)
            if math.gcd(step, n) == 1: break
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
        window_size = self.seq_len + 1
        stride = self.stride
        buf = []

        for i in self._index_permutation(n, rng):
            if (i % consumers) != consumer_id:
                continue
            ex = self.ds[i]
            toks = self._encode_line(ex["text"])
            buf.extend(toks)

            j = 0
            while len(buf) - j >= window_size:
                w = buf[j:j + window_size]
                yield torch.tensor(w, dtype=torch.long)
                j += stride
            if j > 0:
                buf = buf[j:]


# ------------------------- schedule -------------------------
def get_lr(update_idx, warmup_steps, learning_rate, lr_decay_iters, min_lr):
    # Linear warmup
    if update_idx < warmup_steps:
        return learning_rate * (update_idx + 1) / max(1, warmup_steps)
    # After decay iters, return min LR
    if update_idx > lr_decay_iters:
        return min_lr
    # Cosine decay
    decay_ratio = (update_idx - warmup_steps) / max(1, (lr_decay_iters - warmup_steps))
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ------------------------- validate -------------------------
@torch.no_grad()
def validate(model, loader, device, args, max_batches=200):
    model.eval()
    losses = AverageMeter()
    for batch_idx, inputs in enumerate(loader):
        if batch_idx >= max_batches: break
        if inputs.size(1) != args.seq_len + 1:
            continue
        x = inputs[:, :-1].contiguous().to(device, non_blocking=True)
        y = inputs[:, 1:].contiguous().to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', enabled=args.amp):
            attn = torch.ones_like(x, dtype=torch.long)            # <— all tokens attend
            logits = model(x, attention_mask=attn).logits
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T))
        losses.update(loss.item(), B*T)
    losses.all_reduce()
    val_loss = losses.avg
    val_ppl = float(np.exp(np.clip(val_loss, 0, 20)))
    return {'val_loss': val_loss, 'val_ppl': val_ppl}


# ------------------------- train (enhanced with periodic logging) -------------------------
def train_one_epoch(model, dataloader, optimizer, device, scaler, args,
                    epoch, steps_total_start, warmup_steps, lr_decay_iters,
                    val_loader=None):
    """
    Terms:
      - micro-step: one FWD/BWD on a micro-batch (no optimizer update yet)
      - step: one optimizer update (after GA micro-steps)
      - window: the run of steps since the last log (size = log_every_steps; tail can be shorter)

    Behavior:
      - Prints a window snapshot every args.log_every_steps steps.
      - (Optional) Runs a mini validation every args.mini_val_every_steps steps.
      - Epoch summary returns the *current* (tail-window) train_loss/train_ppl,
        plus epoch-averaged timing/throughput and epoch totals.
    """
    model.train()

    def _sync():
        if device.type == 'cuda': torch.cuda.synchronize()

    # ---- epoch totals / averages ----
    micro_time_sum = 0.0; micro_time_n = 0
    step_time_sum  = 0.0; step_time_n  = 0
    tokens = 0

    # ---- window accumulators (reset after each window log) ----
    window_losses = AverageMeter()          # token-weighted CE over the window
    w_micro_sum = 0.0; w_micro_n = 0
    w_step_sum  = 0.0; w_step_n  = 0
    w_tokens = 0
    w_t0 = time.perf_counter()

    def snapshot_window():
        """Compute metrics over the current logging window (or None if empty). All ranks participate."""
        if window_losses.count <= 0:
            return None
        window_losses.all_reduce()
        elapsed = time.perf_counter() - w_t0
        avg_loss = window_losses.avg
        return {
            "train_loss": float(avg_loss),
            "train_ppl":  float(np.exp(np.clip(avg_loss, 0, 20))),
            "lr":         float(lr),
            "micro_step_time_s": w_micro_sum / max(1, w_micro_n),
            "step_time_s":       w_step_sum  / max(1, w_step_n),
            "tok_per_s":         w_tokens / max(1e-6, elapsed),
            "tokens":            int(w_tokens),
            "time_s":            float(elapsed),
            "steps_in_window":   int(w_step_n),
        }

    # ---- prep ----
    _sync()
    t_epoch0 = time.perf_counter()
    steps = 0                 # optimizer updates this epoch
    micro_steps = 0           # micro-batches this epoch
    steps_total_so_far = steps_total_start

    optimizer.zero_grad(set_to_none=True)
    lr = args.learning_rate
    it = iter(dataloader)
    micro_idx = 0

    # GA = max(1, args.gradient_accumulation_steps)
    # updates_per_epoch = args.steps_per_epoch // GA
    # log_enabled = (args.log_every_n_steps > 0 and updates_per_epoch >= args.log_every_n_steps)

    GA = max(1, args.gradient_accumulation_steps)
    steps_per_epoch = args.micro_steps_per_epoch // GA
    log_enabled = (args.log_every_steps > 0 and steps_per_epoch >= args.log_every_steps)


    last_snap = None  # tail (current) window snapshot

    # ---- loop ----
    while True:
        try:
            batch = next(it)
        except StopIteration:
            break
        if batch.size(1) != args.seq_len + 1:
            continue

        if micro_idx == 0:
            _sync(); t_step0 = time.perf_counter()

        # schedule
        lr = get_lr(steps_total_so_far, warmup_steps, args.learning_rate, lr_decay_iters, args.min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # micro-step
        x = batch[:, :-1].contiguous().to(device, non_blocking=True)
        y = batch[:, 1:].contiguous().to(device, non_blocking=True)
        last_micro_of_step = (micro_idx == GA - 1)
        ctx = nullcontext() if last_micro_of_step else model.no_sync()

        with ctx:
            _sync(); t0 = time.perf_counter()
            with torch.amp.autocast(device_type='cuda', enabled=args.amp):
                attn = torch.ones_like(x, dtype=torch.long)
                logits = model(x, attention_mask=attn).logits
                B, T, V = logits.shape
                loss_full = F.cross_entropy(logits.view(B*T, V), y.view(B*T))
                loss = loss_full / GA
                # token-weighted update
                window_losses.update(loss_full.item(), B*T)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            _sync()
            mt = time.perf_counter() - t0
            micro_time_sum += mt; micro_time_n += 1
            w_micro_sum    += mt; w_micro_n    += 1

        n_tok = x.numel()
        tokens   += n_tok
        w_tokens += n_tok
        micro_steps += 1
        micro_idx += 1

        # finish a step (optimizer update)
        if last_micro_of_step:
            _sync()
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer); scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            _sync()

            st = time.perf_counter() - t_step0
            step_time_sum += st; step_time_n += 1
            w_step_sum    += st; w_step_n    += 1

            steps              += 1
            steps_total_so_far += 1
            micro_idx = 0

            # window print + JSON
            if log_enabled and steps % args.log_every_steps == 0:
                snap = snapshot_window()
                last_snap = snap
                if args.rank == 0 and snap is not None:
                    print(
                        f"[{now()}][Epoch {epoch:03d}][Step {steps:04d}/{steps_per_epoch}] "
                        f"loss={snap['train_loss']:.4f} "
                        f"ppl={snap['train_ppl']:.2f} "
                        f"lr={lr:.6f} "
                        f"step_time={snap['step_time_s']:.3f}s "
                        f"micro_time={snap['micro_step_time_s']:.3f}s "
                        f"tok/s={snap['tok_per_s']:.0f} "
                        f"steps_total={steps_total_so_far}",
                        flush=True
                    )
                    if args.json:
                        try:
                            with open(args.json, "r") as f:
                                log = json.load(f)
                            log.setdefault("updates", {})[f"{epoch}_{steps_total_so_far}"] = {
                                "epoch": int(epoch),
                                "steps": int(steps),                    # updates since epoch start
                                "steps_total": int(steps_total_so_far), # cumulative updates
                                "train_loss": float(snap['train_loss']),
                                "train_ppl":  float(snap['train_ppl']),
                                "lr": float(lr),
                                "micro_step_time_s": float(snap['micro_step_time_s']),
                                "step_time_s":       float(snap['step_time_s']),
                                "tok_per_s":         float(snap['tok_per_s']),
                                "tokens":            int(snap['tokens']),
                                "time_s":            float(snap['time_s']),
                                "steps_in_window":   int(snap['steps_in_window']),
                            }
                            save_log(args.json, log)
                        except Exception as e:
                            print(f"[{now()}][Warning] Failed to update JSON log: {e}", flush=True)

                # reset the window accumulators
                window_losses.reset()
                w_micro_sum = 0.0; w_micro_n = 0
                w_step_sum  = 0.0; w_step_n  = 0
                w_tokens = 0
                w_t0 = time.perf_counter()

            # mini-validation (independent cadence; safe at step boundary)
            if (val_loader is not None and
                getattr(args, "mini_val_every_steps", 0) > 0 and
                steps % args.mini_val_every_steps == 0):
                mini = validate(model, val_loader, device, args, max_batches=getattr(args, "mini_val_max_batches", 64))
                model.train()  # return to train mode
                if args.rank == 0:
                    print(
                        f"[{now()}][MiniVal][Epoch {epoch:03d}][Step {steps:04d}] "
                        f"val_loss={mini['val_loss']:.4f} val_ppl={mini['val_ppl']:.2f} "
                        f"(max_batches={getattr(args, 'mini_val_max_batches', 64)})",
                        flush=True
                    )
                    if args.json:
                        try:
                            with open(args.json, "r") as f:
                                log = json.load(f)
                            log.setdefault("updates", {})[f"minival_{epoch}_{steps_total_so_far}"] = {
                                "epoch": int(epoch),
                                "steps": int(steps),
                                "steps_total": int(steps_total_so_far),
                                "mini_val_loss": float(mini['val_loss']),
                                "mini_val_ppl":  float(mini['val_ppl']),
                                "max_batches":   int(getattr(args, "mini_val_max_batches", 64)),
                            }
                            save_log(args.json, log)
                        except Exception as e:
                            print(f"[{now()}][Warning] Failed to update JSON log (mini-val): {e}", flush=True)

        # optional epoch cap by micro-steps
        if args.micro_steps_per_epoch and micro_steps >= args.micro_steps_per_epoch:
            break

    _sync()
    time_epoch_s = time.perf_counter() - t_epoch0
    tok_per_s_epoch = tokens / max(1e-6, time_epoch_s)

    # Tail window snapshot (handles shorter-than-N tail naturally)
    if last_snap is None:
        last_snap = snapshot_window()
    if last_snap is None:  # edge case: no data
        last_snap = {
            "train_loss": float("nan"),
            "train_ppl":  float("nan"),
            "lr":         float(lr),
            "micro_step_time_s": micro_time_sum / max(1, micro_time_n),
            "step_time_s":       step_time_sum  / max(1, step_time_n),
            "tok_per_s":         tok_per_s_epoch,
            "tokens":            int(tokens),
            "time_s":            float(time_epoch_s),
        }

    return {
        # current (tail-window) train metrics for epoch line & JSON
        'train_loss': float(last_snap['train_loss']),
        'train_ppl':  float(last_snap['train_ppl']),

        # LR + epoch totals/averages
        'lr':         float(lr),
        'micro_steps': int(micro_steps),
        'steps':       int(steps),
        'tokens':      int(tokens),
        'micro_step_time_s': float(micro_time_sum / max(1, micro_time_n)),  # epoch avg
        'step_time_s':       float(step_time_sum  / max(1, step_time_n)),   # epoch avg
        'time_epoch_s':      float(time_epoch_s),
        'tok_per_s':         float(tok_per_s_epoch),                         # epoch avg
    }


# ------------------------- DDP setup/teardown -------------------------
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

    dist.init_process_group(
        backend=args.backend,
        init_method="env://",
        rank=args.rank,
        world_size=args.world_size,
        timeout=datetime.timedelta(seconds=300),
    )
    if args.rank == 0:
        print(f"[{now()}][DDP] backend={args.backend} world_size={args.world_size} "
              f"master={args.master_addr}:{args.master_port} iface={args.iface} local_rank={args.local_rank}", flush=True)

def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ------------------------- training driver -------------------------
def train(args):
    device = torch.device(args.device)

    data_root = Path(args.data).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"--data path not found: {data_root}")

    cache_dir = Path(args.cache_dir) if args.cache_dir else (data_root / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)

    # tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 uses eos as pad
    vocab_size = len(tokenizer)

    if args.rank == 0:
        print(f"[{now()}] Loading Parquet dataset from {data_root} (cache: {cache_dir}) ...", flush=True)

    ds_train, ds_val = hf_load_train_val_parquet(
        data_root, val_fraction=args.val_fraction, seed=42, cache_dir=str(cache_dir)
    )

    # datasets/loaders
    train_ds = GPT2Windows(ds_train, tokenizer, seq_len=args.seq_len, stride=args.seq_len,
                           world_size=args.world_size, rank=args.rank, seed=args.seed, append_eos=True)
    val_ds = GPT2Windows(ds_val, tokenizer, seq_len=args.seq_len, stride=max(1, args.seq_len - 256),
                         world_size=args.world_size, rank=args.rank, seed=args.seed + 1, append_eos=True)

    def _seed_worker(worker_id):
        worker_seed = (args.seed + args.rank * max(1, args.workers) + worker_id) % 2**32
        np.random.seed(worker_seed); random.seed(worker_seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
        persistent_workers=(args.workers > 0),
        worker_init_fn=_seed_worker,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=min(2, args.workers),
        pin_memory=True,
        worker_init_fn=_seed_worker,
    )

    # model config (GPT-2 small)
    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=args.seq_len,
        n_ctx=args.seq_len,
        n_embd=768, n_layer=12, n_head=12, n_inner=3072,
        activation_function="gelu",
        resid_pdrop=0.1, attn_pdrop=0.1, embd_pdrop=0.1,
        layer_norm_epsilon=1e-5, initializer_range=0.02,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(cfg).to(device)
    model.config.use_cache = False

    # DDP
    model = DDP(
        model,
        device_ids=[args.local_rank] if device.type == "cuda" else None,
        gradient_as_bucket_view=True,
        find_unused_parameters=False,
        static_graph=args.static_graph,
    )

    if args.rank == 0:
        n_tr = len(ds_train); n_va = len(ds_val)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n[{now()}] DATA ROOT: {data_root}")
        print(f"  Train docs: {n_tr:,} | Val docs: {n_va:,}")
        print(f"  Vocab size: {vocab_size:,} | Seq len: {args.seq_len}")
        print(f"[{now()}] Model: GPT-2 small | Params: {n_params:,}")
        print(f"  LR: {args.learning_rate} | Batch/GPU: {args.batch_size} | GA: {args.gradient_accumulation_steps}")
        print("=" * 60, flush=True)

    # param groups
    decay_params, nodecay_params = [], []
    for p in model.parameters():
        if not p.requires_grad: continue
        (decay_params if p.ndim >= 2 else nodecay_params).append(p)

    optimizer = AdamW(
        [{"params": decay_params, "weight_decay": args.weight_decay},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda" and args.amp))

    # -------- schedule planning (auto) --------
    GA = max(1, args.gradient_accumulation_steps)

    # force micro_steps_per_epoch to a multiple of GA (trim down)
    orig_micro_steps_per_epoch = args.micro_steps_per_epoch
    adj_micro_steps_per_epoch = (orig_micro_steps_per_epoch // GA) * GA
    if adj_micro_steps_per_epoch <= 0:
        raise ValueError("micro_steps_per_epoch must be at least gradient_accumulation_steps.")
    if args.rank == 0 and adj_micro_steps_per_epoch != orig_micro_steps_per_epoch:
        print(
            f"[{now()}][Note] micro_steps_per_epoch adjusted from {orig_micro_steps_per_epoch} "
            f"to {adj_micro_steps_per_epoch} to be a multiple of GA={GA}.",
            flush=True
        )
    args.micro_steps_per_epoch = adj_micro_steps_per_epoch  # keep arg as micro-steps/epoch

    # Optimizer steps per epoch & total planned optimizer steps
    steps_per_epoch = args.micro_steps_per_epoch // GA
    total_steps_planned = args.epochs * steps_per_epoch

    # resolve LR schedule (auto unless explicitly set)
    lr_decay_iters = args.lr_decay_iters if args.lr_decay_iters > 0 else total_steps_planned
    warmup_steps = args.warmup_steps if args.warmup_steps >= 0 else min(1000, max(1, total_steps_planned // 10))

    if args.rank == 0:
        print(
            f"[{now()}] Plan: epochs={args.epochs}, micro_steps/epoch={args.micro_steps_per_epoch} "
            f"(GA={GA} → steps/epoch={steps_per_epoch}), total_steps_planned={total_steps_planned}"
        )
        print(f"[{now()}] LR schedule: warmup {warmup_steps} steps, cosine decay to {lr_decay_iters} (min_lr={args.min_lr}).")

        # Window logging info
        if args.log_every_steps > 0:
            if steps_per_epoch >= args.log_every_steps:
                n_logs_per_epoch = steps_per_epoch // args.log_every_steps
                print(f"[{now()}] Periodic logging: every {args.log_every_steps} steps (~{n_logs_per_epoch} times per epoch)")
            else:
                print(f"[{now()}] Periodic logging: disabled (epoch too short: {steps_per_epoch} < {args.log_every_steps})")
        else:
            print(f"[{now()}] Periodic logging: disabled")

    # init JSON log
    if args.rank == 0:
        if args.json is None: args.json = "gpt2_training.json"
        log = {
            "time": now(),
            "data_root": str(data_root),
            "cache_dir": str(cache_dir),
            "config": vars(args),
            "plan": {
                "ga": GA,
                "micro_steps_per_epoch_original": int(orig_micro_steps_per_epoch),
                "micro_steps_per_epoch": int(args.micro_steps_per_epoch),
                "steps_per_epoch": int(steps_per_epoch),             # optimizer steps per epoch
                "total_steps_planned": int(total_steps_planned),     # total optimizer steps planned
                "lr_decay_iters": int(lr_decay_iters),
                "warmup_steps": int(warmup_steps),
            },
            "vocab_size": vocab_size,
            "epochs": {},
            "updates": {},  # window logs keyed during training
        }
        save_log(args.json, log)
        print(f"[{now()}] Logging to {args.json}")

    best_ppl = float('inf')
    steps_total = 0  # cumulative optimizer steps so far

    for epoch in range(args.epochs):
        if args.rank == 0:
            print(f"[{now()}][Epoch {epoch:03d}] Start", flush=True)
        train_ds.set_epoch(epoch)

        # train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, scaler, args,
            epoch, steps_total, warmup_steps, lr_decay_iters, val_loader
        )
        steps_total += train_metrics['steps']  # accumulate optimizer steps

        # validate full (only at end of epoch)
        val_metrics = validate(model, val_loader, device, args, max_batches=args.val_max_batches)
        current_lr = optimizer.param_groups[0]['lr']

        # stdout per-epoch
        if args.rank == 0:
            print(
                f"[{now()}][Epoch {epoch:03d}] "
                f"micro_steps={train_metrics['micro_steps']} "
                f"steps={train_metrics['steps']} "
                f"loss={train_metrics['train_loss']:.4f} "
                f"ppl={train_metrics['train_ppl']:.2f} "
                f"val_ppl={val_metrics['val_ppl']:.2f} "
                f"step_time={train_metrics['step_time_s']:.3f}s "
                f"micro_time={train_metrics['micro_step_time_s']:.3f}s "
                f"tok/s={train_metrics['tok_per_s']:.0f} "
                f"steps_total={steps_total}",
                flush=True
            )

            # JSON epoch log
            epoch_metrics = {
                "train_loss": float(train_metrics['train_loss']),
                "train_ppl":  float(train_metrics['train_ppl']),
                "val_loss":   float(val_metrics['val_loss']),
                "val_ppl":    float(val_metrics['val_ppl']),
                "lr":         float(current_lr),

                "micro_steps": int(train_metrics['micro_steps']),
                "steps":       int(train_metrics['steps']),        # optimizer steps this epoch
                "steps_total": int(steps_total),                   # cumulative optimizer steps
                "tokens":      int(train_metrics['tokens']),

                "micro_step_time_s": float(train_metrics['micro_step_time_s']),
                "step_time_s":       float(train_metrics['step_time_s']),
                "time_epoch_s":      float(train_metrics['time_epoch_s']),
                "tok_per_s":         float(train_metrics['tok_per_s']),
            }
            with open(args.json, "r") as f:
                log = json.load(f)
            log["epochs"][str(epoch)] = epoch_metrics
            save_log(args.json, log)

            if val_metrics['val_ppl'] < best_ppl:
                best_ppl = val_metrics['val_ppl']
                print(f"[{now()}] New best validation perplexity: {best_ppl:.2f}", flush=True)

    if args.rank == 0:
        print(f"\n[{now()}] Training complete. Best val ppl: {best_ppl:.2f}")


# ------------------------- main -------------------------
def main():
    parser = argparse.ArgumentParser(description='GPT-2 DDP on OpenWebText (with periodic update logging)')
    # DDP/System
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=6)
    parser.add_argument('--iface', type=str, default='ens4f0')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1')
    parser.add_argument('--master_port', type=int, default=29500)
    parser.add_argument('--backend', type=str, default='gloo', choices=['nccl', 'gloo'])
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--static_graph', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--json', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1337)

    # Data
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--val_fraction', type=float, default=0.0005)
    parser.add_argument('--tokenizer', type=str, default='gpt2')

    # Training (choose these; schedule is auto-computed)
    parser.add_argument('--epochs', type=int, default=12)
    # parser.add_argument('--steps_per_epoch', type=int, default=6000)  # micro-steps per epoch (will be trimmed to multiple of GA)
    parser.add_argument('--micro_steps_per_epoch', '--steps_per_epoch', dest='micro_steps_per_epoch', type=int, default=6000,
                        help='Micro-batches per epoch (alias: --steps_per_epoch). Optimizer steps/epoch ~= this / GA.')
    
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=6e-4)
    parser.add_argument('--min_lr', type=float, default=6e-5)
    parser.add_argument('--warmup_steps', '--warmup_optimizer_steps', dest='warmup_steps', type=int, default=-1,
        help='Warmup in OPTIMIZER steps (not micro-steps). -1 = auto (10% of total optimizer steps, capped at 1000).'
    )
    parser.add_argument('--lr_decay_iters', type=int, default=-1, help='-1 = auto (total planned optimizer steps)')
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--val_max_batches', type=int, default=200)
    parser.add_argument('--mini_val_every_steps', type=int, default=0, help='Run a small validation every N optimizer steps. 0=off.')
    parser.add_argument('--mini_val_max_batches', type=int, default=64, help='Batches to use for mini validation.')
    # Model
    parser.add_argument('--seq_len', type=int, default=1024)

    # Periodic logging (NEW - single argument)
    parser.add_argument('--log_every_steps', type=int, default=0, help='Log every N optimizer updates during training. 0 = disabled. '
                            'Automatically disabled if epoch has fewer than N steps.')

    args = parser.parse_args()

    # Determinism & CUDA opts
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    else:
        torch.backends.cudnn.benchmark = True

    if args.device == 'cuda' and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        print(f"[{now()}][Info] Using device=cpu because CUDA is not available", flush=True)
    if args.amp and args.device == 'cpu':
        args.amp = False
        print(f"[{now()}][Info] Disabling AMP because CUDA is not available", flush=True)
    if args.workers < 0:
        args.workers = 0

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    setup_ddp(args)
    if args.rank == 0:
        print(f"[{now()}] Configuration:\n{json.dumps(vars(args), indent=2)}")
    try:
        train(args)
    finally:
        cleanup()


if __name__ == "__main__":
    main()