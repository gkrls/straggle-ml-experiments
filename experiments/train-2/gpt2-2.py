#!/usr/bin/env python3
"""
GPT-2 (124M) DDP trainer on local Parquet OpenWebText.
"""

import os, sys, argparse, time, datetime, json, math, random, warnings, logging, re
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

# from straggle_sim import SlowWorkerPattern

import dpa


# ------------------------- utilities -------------------------
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
    def reset(self): 
        self.sum = 0.0; 
        self.count = 0.0; 
        self.avg = 0.0
        # self.min = 0.0
        # self.max = 0.0
    def update(self, val, n=1): 
        self.sum += float(val)*n
        self.count += n
        self.avg = self.sum / max(1.0, self.count)
        # self.min = min(self.min, val)
        # self.max = max(self.max, val)
    def all_reduce(self):
        if dist.is_available() and dist.is_initialized():
            backend = dist.get_backend()
            device = torch.device(f"cuda:{torch.cuda.current_device()}") if backend == dist.Backend.NCCL else torch.device("cpu")
            t = torch.tensor([self.sum, self.count], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            self.sum, self.count = t.cpu().tolist()
            self.avg = self.sum / max(1.0, self.count)


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
            if (i % consumers) != consumer_id: continue
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
    if update_idx < warmup_steps: return learning_rate * (update_idx + 1) / max(1, warmup_steps)
    # After decay iters, return min LR
    if update_idx > lr_decay_iters: return min_lr
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
    step_start = time.perf_counter()
    for batch_idx, inputs in enumerate(loader):
        if batch_idx >= max_batches: break
        if inputs.size(1) != args.seq_len + 1: continue
        x = inputs[:, :-1].contiguous().to(device, non_blocking=True)
        y = inputs[:, 1:].contiguous().to(device, non_blocking=True)
        with torch.amp.autocast(device_type='cuda', enabled=args.amp):
            attn = torch.ones_like(x, dtype=torch.long)            # <— all tokens attend
            # logits = model(x, attention_mask=attn).logits
            logits = model(x, attention_mask=attn)[0]
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B*T, V), y.view(B*T))
        losses.update(loss.item(), B*T)
    # losses.all_reduce()
    val_loss = losses.avg
    val_ppl = float(np.exp(np.clip(val_loss, 0, 20)))
    return {'loss': val_loss, 'ppl': val_ppl, 'time': time.perf_counter() - step_start}


# ------------------------- train (enhanced with periodic logging) -------------------------
def train_one_epoch(model, dataloader, optimizer, device, scaler, args,
                    epoch, global_step_start, warmup_steps, lr_decay_iters,
                    val_loader=None):
    model.train()
    use_cuda = (device.type == "cuda")
    def _sync():
        if use_cuda: torch.cuda.synchronize()

    GA = max(1, args.gradient_accumulation_steps)
    steps_per_epoch = max(1, args.micro_steps_per_epoch // GA)
    log_every = max(0, args.log_every_steps)

    # ----- epoch counters -----
    epoch_t0 = time.perf_counter()
    micro_count = 0
    step_count  = 0
    tokens      = 0

    # epoch time accumulators (+ min/max)
    step_time = 0.0
    micro_time = 0.0
    step_time_min, step_time_max = float("inf"), 0.0
    micro_time_min, micro_time_max = float("inf"), 0.0

    # ----- window accumulators (reset each log) -----
    w_start = time.perf_counter()
    w_step_time = 0.0
    w_step_count = 0
    w_micro_time = 0.0
    w_tokens = 0
    w_loss_sum = 0.0      # token-weighted CE
    w_token_count = 0

    last_train_loss = float("nan")

    optimizer.zero_grad(set_to_none=True)
    lr = args.learning_rate
    micros_in_step = 0
    step_t0 = None

    it = iter(dataloader)
    while True:
        try:
            batch = next(it)
        except StopIteration:
            break
        if batch.size(1) != args.seq_len + 1:
            continue

        # start a new optimizer step timing on first micro
        if micros_in_step == 0:
            _sync()
            step_t0 = time.perf_counter()
            global_step = global_step_start + step_count
            lr = get_lr(global_step, warmup_steps, args.learning_rate, lr_decay_iters, args.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        # ----- micro-step -----
        x = batch[:, :-1].contiguous().to(device, non_blocking=True)
        y = batch[:, 1:].contiguous().to(device, non_blocking=True)
        is_last_micro = (micros_in_step == GA - 1)
        ctx = nullcontext() if is_last_micro else model.no_sync()

        _sync(); t0 = time.perf_counter()
        with ctx:
            with torch.amp.autocast(device_type="cuda", enabled=args.amp):
                attn = torch.ones_like(x, dtype=torch.long)
                # logits = model(x, attention_mask=attn).logits
                logits = model(x, attention_mask=attn)[0]
                B, T, V = logits.shape
                loss_full = F.cross_entropy(logits.view(B*T, V), y.view(B*T))
                loss = loss_full / GA
            if scaler is not None: scaler.scale(loss).backward()
            else: loss.backward()
        _sync()
        micro_dt = time.perf_counter() - t0

        # micro accounting
        tok = x.numel()
        tokens += tok
        w_tokens += tok
        w_loss_sum += float(loss_full.item()) * (B * T)
        w_token_count += (B * T)

        micro_count += 1
        micros_in_step += 1

        micro_time += micro_dt
        w_micro_time += micro_dt
        micro_time_min = min(micro_time_min, micro_dt)
        micro_time_max = max(micro_time_max, micro_dt)

        # ----- finish GA group → optimizer step -----
        if is_last_micro:
            _sync()
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            _sync()

            step_dt = time.perf_counter() - step_t0
            step_count += 1
            global_step = global_step_start + step_count

            step_time += step_dt
            w_step_time += step_dt
            w_step_count += 1
            step_time_min = min(step_time_min, step_dt)
            step_time_max = max(step_time_max, step_dt)

            micros_in_step = 0

            # ----- periodic window log -----
            if log_every > 0 and step_count % log_every == 0:
                elapsed = max(1e-6, time.perf_counter() - w_start)
                train_loss = (w_loss_sum / max(1, w_token_count)) if w_token_count > 0 else float("nan")
                last_train_loss = train_loss
                train_ppl = float(np.exp(np.clip(train_loss, 0, 20))) if math.isfinite(train_loss) else float("nan")

                # window avgs
                step_time_win = w_step_time / max(1, w_step_count)
                micro_time_win = w_micro_time / max(1, w_step_count * GA)  # per-micro avg
                tok_per_s = w_tokens / elapsed

                # if args.rank == 0:
                print(f"[{now()}][Epoch {epoch:03d}][Step {step_count:04d}/{steps_per_epoch}] global_step={global_step} "
                      f"train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} lr={lr:.6f} "
                      f"step_time={step_time_win:.3f}s step_time_min={step_time_min:.3f} step_time_max={step_time_max:.3f} "
                      f"micro_time={micro_time_win:.3f}s tp={tok_per_s:.0f} tok/s",flush=True)
                if args.json:
                    try:
                        with open(args.json, "r") as f:
                            log = json.load(f)
                        updates = log.setdefault("updates", {})
                        ep = updates.setdefault(str(epoch), {})
                        ep[f"{step_count:04d}"] = {
                            "global_step": int(global_step),
                            "train_loss": float(train_loss),
                            "train_ppl":  float(train_ppl),
                            "lr": float(lr),
                            "step_time":       float(step_time_win),
                            "micro_step_time": float(micro_time_win),
                            "throughput":      float(tok_per_s),
                            "tokens":          int(w_tokens),
                        }
                        save_log(args.json, log)
                    except Exception as e:
                        print(f"[{now()}][Warning] Failed to update JSON log: {e}", flush=True)

                # reset window
                w_start = time.perf_counter()
                w_step_time = 0.0; w_step_count = 0
                w_micro_time = 0.0
                w_tokens = 0
                w_loss_sum = 0.0; w_token_count = 0

            # ----- optional mini validation -----
            if (val_loader is not None and getattr(args, "mini_val_every_steps", 0) > 0
                    and step_count % args.mini_val_every_steps == 0):
                val_metrics = validate(model, val_loader, device, args, max_batches=getattr(args, "mini_val_max_batches", 64))
                model.train()
                # if args.rank == 0:
                print(f"[{now()}][MiniVal][Epoch {epoch:03d}][Step {step_count:04d}] "
                      f"val_loss={val_metrics['loss']:.4f} val_ppl={val_metrics['ppl']:.2f} val_time={val_metrics['time']:.2f}s",
                      flush=True)
                if args.json:
                    try:
                        with open(args.json, "r") as f:
                            log = json.load(f)
                            updates_minival = log.setdefault("minival", {})
                            epm = updates_minival.setdefault(str(epoch), {})
                            epm[f"{step_count:04d}"] = {
                                "global_step": int(global_step),
                                "mini_val_loss": float(val_metrics['loss']),
                                "mini_val_ppl":  float(val_metrics['ppl']),
                                "max_batches":   int(getattr(args, "mini_val_max_batches", 64)),
                            }
                        save_log(args.json, log)
                    except Exception as e:
                        print(f"[{now()}][Warning] Failed to update JSON log (mini-val): {e}", flush=True)

        # optional epoch cap by micro-steps
        if args.micro_steps_per_epoch and micro_count >= args.micro_steps_per_epoch:
            break

    # ----- epoch summary -----
    _sync()
    time_epoch_s = time.perf_counter() - epoch_t0
    tok_per_s = tokens / max(1e-6, time_epoch_s)

    if not math.isfinite(last_train_loss):
        last_train_loss = w_loss_sum / w_token_count if w_token_count > 0 else float("nan")

    train_ppl = float(np.exp(np.clip(last_train_loss, 0, 20))) if math.isfinite(last_train_loss) else float("nan")

    return {
        "loss": float(last_train_loss),
        "ppl":  float(train_ppl),
        "lr":         float(lr),

        "micro_steps": int(micro_count),
        "steps":       int(step_count),
        "tokens":      int(tokens),

        "micro_step_time":     float(micro_time / max(1, micro_count)),  # epoch avg
        "micro_step_time_min": float(micro_time_min if micro_count > 0 else float("nan")),
        "micro_step_time_max": float(micro_time_max if micro_count > 0 else float("nan")),
        "step_time":           float(step_time / max(1, step_count)),    # epoch avg
        "step_time_min":       float(step_time_min if step_count > 0 else float("nan")),
        "step_time_max":       float(step_time_max if step_count > 0 else float("nan")),
        "epoch_time":          float(time_epoch_s),

        "throughput":    float(tok_per_s),
    }



# ------------------------- training driver -------------------------

class SimpleDDP(DDP):
    def __init__(self, module, **kwargs):
        super().__init__(module, broadcast_buffers=False, **kwargs)
        self.require_forward_param_sync = False
    
    def _sync_params(self):
        pass  # Never sync parameters
    
    def _distributed_broadcast_coalesced(self, *args, **kwargs):
        pass  # Never broadcast anything
    
    # Optionally override forward to skip the check entirely
    def forward(self, *inputs, **kwargs):
        # Skip all DDP's sync logic, just call the module
        return self.module(*inputs, **kwargs)

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

    # Model config (GPT-2 small)
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
        return_dict=False # IMPORTANT for straggle sim
    )
    model = GPT2LMHeadModel(cfg).to(device)
    model.config.use_cache = False

    # DDP
    # model = DDP(model, device_ids=[args.local_rank] if device.type == "cuda" else None,
    #             bucket_cap_mb=args.bucket_cap_mb, gradient_as_bucket_view=True, find_unused_parameters=False, static_graph=args.static_graph)
    model = DDP(model, device_ids=[args.local_rank] if device.type == "cuda" else None, broadcast_buffers=False, bucket_cap_mb=args.bucket_cap_mb, 
                gradient_as_bucket_view=True, find_unused_parameters=False, static_graph=False)
    model.require_forward_param_sync = False

    # Wrap the model if DPA backend is requested
    if args.backend.startswith("dpa"):
        model = dpa.DDPWrapper(model, straggle = args.straggle_k if args.straggle_k > 0 else args.world_size, prescale=args.prescale)

    # Straggle sim
    # straggle = dpa.DDPStraggleSim(points=args.straggle_points, prob=args.straggle_prob, amount=args.straggle_amount, ranks=args.straggle_ranks)

    straggle = dpa.DDPStraggleSim(points=args.straggle_points, prob=args.straggle_prob, amount=args.straggle_amount, ranks=args.straggle_ranks, 
                                  multiplier_range=args.straggle_multiply, verbose=args.straggle_verbose)        
    if straggle.attach(model): print(f"{straggle} created and active for rank {args.rank}")
    else: print(f"{straggle} created but inactive for rank {args.rank}")
    # straggle_sim = SlowWorkerPattern(points=args.straggle_points, prob=args.straggle_prob, amount=args.straggle_amount,
    #                                 ranks=args.straggle_ranks, multiplier_range=args.straggle_multiply, seed=42,
    #                                 verbose=args.straggle_verbose)
    # if straggle_sim.attach(model): print(f"Straggle sim initialized with {straggle_sim}")
    # else: print(f"Straggle sim inactive")

    # if args.rank == 0:
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

    optimizer = AdamW([{"params": decay_params, "weight_decay": args.weight_decay}, {"params": nodecay_params, "weight_decay": 0.0}], 
                      lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8)
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

    # if args.rank == 0:
    print(f"[{now()}] Plan: epochs={args.epochs}, micro_steps/epoch={args.micro_steps_per_epoch} "
          f"(GA={GA} → steps/epoch={steps_per_epoch}), total_steps_planned={total_steps_planned}")
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
    # if args.rank == 0:
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
    global_step = 0  # cumulative optimizer steps so far

    for epoch in range(args.epochs):
        print(f"[{now()}][Epoch {epoch:03d}] ...", flush=True)
        
        epoch_start = time.time()

        straggle.reset_stats()
        
        train_ds.set_epoch(epoch)

        # train
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, scaler, args, epoch, global_step, warmup_steps, lr_decay_iters, val_loader)
        global_step += train_metrics['steps']  # accumulate optimizer steps

        # validate full (only at end of epoch)
        val_metrics = validate(model, val_loader, device, args, max_batches=args.val_max_batches)
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start
        # stdout per-epoch
        # if args.rank == 0:
        print(
            f"[{now()}][Epoch {epoch:03d}] "
            f"global_step={global_step} ",
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_ppl={train_metrics['ppl']:.2f} "
            f"val_ppl={val_metrics['ppl']:.2f} "
            f"micro_steps={train_metrics['micro_steps']} "
            f"micro_time={train_metrics['micro_step_time']:.3f}s "
            f"steps={train_metrics['steps']} "
            f"step_time={train_metrics['step_time']:.3f}s "
            f"epoch_train_time={train_metrics['epoch_time']:.3f}s ",
            f"epoch_time={epoch_time:.3f}s "
            f"tp={train_metrics['throughput']:.0f} tok/s "
            f"straggle_events={straggle.get_stats()['num_straggle_events']}", flush=True
        )

        # JSON epoch log
        epoch_metrics = {
            "train_loss": float(train_metrics['loss']),
            "train_ppl":  float(train_metrics['ppl']),
            "val_loss":   float(val_metrics['loss']),
            "val_ppl":    float(val_metrics['ppl']),
            "lr":         float(current_lr),

            "micro_steps": int(train_metrics['micro_steps']),
            "steps":       int(train_metrics['steps']),        # optimizer steps this epoch
            "global_step": int(global_step),                   # cumulative optimizer steps
            "tokens":      int(train_metrics['tokens']),

            "micro_step_time":        float(train_metrics['micro_step_time']),
            "micro_step_time_min":    float(train_metrics['micro_step_time_min']),
            "micro_step_time_max":    float(train_metrics['micro_step_time_max']),
            "step_time":              float(train_metrics['step_time']),
            "step_time_min":          float(train_metrics['step_time_min']),
            "step_time_max":          float(train_metrics['step_time_max']),
            "epoch_time":             float(epoch_time),
            "epoch_train_time":       float(train_metrics['epoch_time']),
            "epoch_train_throughput": float(train_metrics['throughput']),

            # straggle-sim
            "straggle" : straggle.get_stats() if straggle.active else {}
        }
        with open(args.json, "r") as f:
            log = json.load(f)
        log["epochs"][str(epoch)] = epoch_metrics
        save_log(args.json, log)

        if val_metrics['ppl'] < best_ppl:
            best_ppl = val_metrics['ppl']
            print(f"[{now()}] New best validation perplexity: {best_ppl:.2f}", flush=True)

    if args.rank == 0:
        print(f"\n[{now()}] Training complete. Best val ppl: {best_ppl:.2f}")


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

    if args.device == 'cuda' and torch.cuda.is_available(): torch.cuda.set_device(args.local_rank)

    os.environ.setdefault("RANK", str(args.rank))
    os.environ.setdefault("WORLD_SIZE", str(args.world_size))
    os.environ.setdefault("MASTER_ADDR", args.master_addr)
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", args.iface)
    os.environ.setdefault("NCCL_SOCKET_IFNAME", args.iface)

    # Initialize process group
    if args.backend.startswith("dpa"):
        if not args.dpa_conf: raise RuntimeError(f"--dpa_conf required for backend {args.backend}")
        dpa_device = dpa.DPADeviceOptions.from_config(args.dpa_conf)
        dpa_backend = dpa.DPADpdkBackendOptions.from_config(args.dpa_conf)
        pg_options = dpa.ProcessGroupDPADpdkOptions(dpa_device, dpa_backend)
        # pg_options.hint_pinned_tensor_size = max(200_000_000, args.bucket_cap_mb * (2 ** 20) * 4) # observed max around 150-is MB
        # pg_options.hint_pinned_tensor_pool_size = 20                                              # observed count 13
        pg_options.hint_pinned_tensor_size = max(200_000_000, args.bucket_cap_mb * (2 ** 20) * 4 if args.bucket_cap_mb is not None else 0) # observed max around 150-is MB
        pg_options.hint_pinned_tensor_pool_size = 20                                                                                       # observed count 13
        dist.init_process_group(backend=args.backend, init_method="env://", rank=args.rank, world_size=args.world_size, timeout = datetime.timedelta(seconds=60), pg_options=pg_options)
    else:
        dist.init_process_group(backend=args.backend, init_method="env://", rank=args.rank, world_size=args.world_size, timeout=datetime.timedelta(seconds=60))

    print(f"[{now()}][DDP] backend={args.backend} world_size={args.world_size} "
          f"master={args.master_addr}:{args.master_port} iface={args.iface} local_rank={args.local_rank}", flush=True)

# ------------------------- main -------------------------
def main():
    parser = argparse.ArgumentParser(description='GPT-2 DDP on OpenWebText (with periodic update logging)')

    # DDP/System
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=6)
    parser.add_argument('--iface', type=str, default='ens4f0')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1')
    parser.add_argument('--master_port', type=int, default=29500)
    parser.add_argument('--backend', type=str, default='gloo', choices=['nccl', 'gloo', 'dpa_dpdk'])
    parser.add_argument("--dpa_conf", type=str, default=None, help="Path to dpa config.json")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--static_graph', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--json', type=str, default="gpt2.json", help="Path to JSON run log")
    parser.add_argument('--log_every_steps', type=int, default=0, help='Log every N optimizer updates during training. 0 = disabled. '
                            'Automatically disabled if epoch has fewer than N steps.')

    # Data
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--val_fraction', type=float, default=0.0005)
    parser.add_argument('--tokenizer', type=str, default='gpt2')

    # Training
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--micro_steps_per_epoch', '--steps_per_epoch', dest='micro_steps_per_epoch', type=int, default=6000,
                        help='Micro-batches per epoch (alias: --steps_per_epoch). Optimizer steps/epoch ~= this / GA.')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=6e-4)
    parser.add_argument('--min_lr', type=float, default=6e-5)
    parser.add_argument('--warmup_steps', '--warmup_optimizer_steps', dest='warmup_steps', type=int, default=-1,
                        help='Warmup in OPTIMIZER steps (not micro-steps). -1 = auto (10% of total optimizer steps, capped at 1000).')
    parser.add_argument('--lr_decay_iters', type=int, default=-1, help='-1 = auto (total planned optimizer steps)')
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--val_max_batches', type=int, default=200)
    parser.add_argument('--mini_val_every_steps', type=int, default=0, help='Run a small validation every N optimizer steps. 0=off.')
    parser.add_argument('--mini_val_max_batches', type=int, default=64, help='Batches to use for mini validation.')

    parser.add_argument('--prescale', action="store_true", help="Prescale gradients for allreduce")
    parser.add_argument("--bucket_cap_mb", type=int, default=None, help="DDP bucket capacity")

    # Model config -- mostly fixed
    parser.add_argument('--seq_len', type=int, default=1024)

    # Straggle
    def csv_ints(s: str) -> list[int]:
        if not s: return []
        try: return [int(x) for x in re.split(r"\s*,\s*", s) if x]
        except ValueError: raise argparse.ArgumentTypeError("Expected a comma-separated list of integers (e.g. 1,2,3)")
    parser.add_argument("--straggle_points", type=int, help="Number of straggle points (1-3). Use 0 for no straggle sim", default=0)
    parser.add_argument("--straggle_prob", type=float, help="Probability to straggle at each point", default=0)
    parser.add_argument("--straggle_ranks", type=csv_ints, help="comma separated list of ints", default=[])
    parser.add_argument("--straggle_amount", type=float, help="base straggle amount in seconds (e.g. mean step time)", default=0)
    parser.add_argument("--straggle_multiply", type=float, nargs=2, metavar=("lo","hi"), help="straggle amount multipler lo and hi", default=[1.0, 1.0])
    parser.add_argument("--straggle_verbose", action='store_true')
    parser.add_argument("--straggle_k", type=int, default=0)



    args = parser.parse_args()
    args.seed = args.seed + args.rank * 1000

    if args.straggle_k:
        print(f"!! Straggler mitigation ENABLED with straggle_k={args.straggle_k} !!")
    else:
        print(f"!! Straggler mitigation ENABLED !!")

    # Determinism & CUDA opts
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

    if args.device == 'cuda' and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        print(f"[{now()}][Info] Using device=cpu because CUDA is not available", flush=True)
    if args.amp and args.device == 'cpu':
        args.amp = False
        print(f"[{now()}][Info] Disabling AMP because CUDA is not available", flush=True)

    args.workers = max(args.workers, 0)


    sys.stdout.reconfigure(line_buffering=True)

    setup_ddp(args)
    print(f"[{now()}] Configuration:\n{json.dumps(vars(args), indent=2)}")

    try:
        train(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()