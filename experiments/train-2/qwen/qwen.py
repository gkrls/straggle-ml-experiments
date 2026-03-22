#!/usr/bin/env python3
"""
Qwen2.5-0.5B (or any HF causal LM) instruction fine-tuning on Alpaca.
"""

import os, sys, argparse, time, datetime, json, math, random, warnings, re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoConfig,
    get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup,
)
from datasets import load_dataset, DownloadMode

try:
    import dpa
except Exception:
    dpa = None


# ------------------------- utilities -------------------------
def now():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def save_log(path, log):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2)
    os.replace(tmp, path)

class AverageMeter:
    def __init__(self): self.reset()
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


# ------------------------- Alpaca prompt formatting -------------------------
ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{output}"
)

ALPACA_PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

def format_alpaca(example):
    """Format a single Alpaca example into a single string."""
    if example.get("input", "").strip():
        return ALPACA_PROMPT_WITH_INPUT.format(**example)
    else:
        return ALPACA_PROMPT_NO_INPUT.format(**example)


# ------------------------- dataset -------------------------
class AlpacaSFTDataset(Dataset):
    """
    Pre-tokenized Alpaca instruction-tuning dataset.
    Each item is a (seq_len+1,) tensor for causal-LM style x=tokens[:-1], y=tokens[1:].
    Optionally masks the prompt portion of the loss (only trains on response tokens).
    """
    def __init__(self, examples, tokenizer, max_seq_len=512, mask_prompt=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_prompt = mask_prompt
        self.data = []
        self.response_marker = tokenizer.encode("\n### Response:\n", add_special_tokens=False)

        skipped = 0
        for ex in examples:
            text = format_alpaca(ex)
            ids = tokenizer.encode(text, add_special_tokens=True)
            if tokenizer.eos_token_id is not None and (len(ids) == 0 or ids[-1] != tokenizer.eos_token_id):
                ids.append(tokenizer.eos_token_id)
            if len(ids) < 4:
                skipped += 1
                continue
            # truncate to max_seq_len + 1 (need +1 for shifted targets)
            ids = ids[:max_seq_len + 1]
            self.data.append(ids)

        if skipped > 0:
            print(f"[{now()}] AlpacaSFTDataset: skipped {skipped} examples (too short)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids = self.data[idx]
        # pad to max_seq_len + 1
        pad_len = (self.max_seq_len + 1) - len(ids)
        attention_len = len(ids)

        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            ids = ids + [pad_id] * pad_len

        input_ids = torch.tensor(ids, dtype=torch.long)

        # build labels: -100 for padding (and optionally prompt)
        labels = input_ids.clone()
        labels[attention_len:] = -100  # mask padding

        if self.mask_prompt:
            # find the response marker in the token ids and mask everything before it
            resp_start = _find_sublist(ids, self.response_marker)
            if resp_start >= 0:
                # mask up to and including the marker
                mask_end = resp_start + len(self.response_marker)
                labels[:mask_end] = -100

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_len": attention_len,  # for building attention mask
        }


def _find_sublist(lst, sub):
    """Find first occurrence of sub in lst. Returns index or -1."""
    n = len(sub)
    for i in range(len(lst) - n + 1):
        if lst[i:i+n] == sub:
            return i
    return -1


def _collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    labels    = torch.stack([b["labels"] for b in batch])
    # build attention mask from attention_len
    bsz, seq = input_ids.shape
    attention_mask = torch.zeros(bsz, seq, dtype=torch.long)
    for i, b in enumerate(batch):
        attention_mask[i, :b["attention_len"]] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# ------------------------- schedule -------------------------
# (used only when --sched manual; otherwise HF schedulers)
def get_lr(update_idx, warmup_steps, learning_rate, lr_decay_iters, min_lr):
    if update_idx < warmup_steps:
        return learning_rate * (update_idx + 1) / max(1, warmup_steps)
    if update_idx > lr_decay_iters:
        return min_lr
    decay_ratio = (update_idx - warmup_steps) / max(1, (lr_decay_iters - warmup_steps))
    decay_ratio = min(max(decay_ratio, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ------------------------- validate -------------------------
@torch.no_grad()
def validate(model, loader, device, args, max_batches=0):
    """Per-rank validation. Computes loss and perplexity. max_batches=0 means all."""
    model.eval()
    losses = AverageMeter()
    token_losses = AverageMeter()
    val_start = time.perf_counter()

    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        labels         = batch["labels"].to(device, non_blocking=True)
        attention_mask  = batch["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=args.amp):
            # Same manual loss as training — avoid HF's logits.float() OOM (see training loop comment)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
            shift_labels = labels[:, 1:].contiguous().view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

        # count non-masked tokens for proper perplexity
        n_tokens = (labels != -100).sum().item()
        losses.update(loss.item(), input_ids.size(0))
        token_losses.update(loss.item(), max(1, n_tokens))

    val_loss = losses.avg
    val_ppl = float(np.exp(np.clip(val_loss, 0, 20)))
    return {
        "loss": float(val_loss),
        "ppl":  float(val_ppl),
        "time": float(time.perf_counter() - val_start),
    }


# ------------------------- train one epoch -------------------------
def train_one_epoch(model, dataloader, optimizer, sched, device, scaler, args,
                    epoch, global_step_start, val_loader=None, log=None):
    model.train()
    use_cuda = (device.type == "cuda")
    def _sync():
        if use_cuda: torch.cuda.synchronize()

    GA = max(1, args.gradient_accumulation_steps)
    total_batches = len(dataloader)
    steps_per_epoch = max(1, total_batches // GA)
    log_every = max(0, args.log_every_opt_steps)

    # ----- epoch counters -----
    epoch_t0 = time.perf_counter()
    micro_count = 0
    step_count  = 0
    tokens      = 0

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
    w_loss_sum = 0.0
    w_token_count = 0

    last_train_loss = float("nan")

    optimizer.zero_grad(set_to_none=True)
    micros_in_step = 0
    step_t0 = None

    for batch_idx, batch in enumerate(dataloader, 1):
        if micros_in_step == 0:
            _sync()
            step_t0 = time.perf_counter()

        # ----- micro-step -----
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        labels         = batch["labels"].to(device, non_blocking=True)
        attention_mask  = batch["attention_mask"].to(device, non_blocking=True)

        is_last_micro = (micros_in_step == GA - 1)
        ctx = nullcontext() if (GA == 1 or is_last_micro) else model.no_sync()

        _sync()
        t0 = time.perf_counter()
        with ctx:
            with torch.amp.autocast(device_type="cuda", enabled=args.amp):
                # NOTE: We do NOT pass labels= to the model. HuggingFace internally does
                # logits.float() which allocates a full FP32 copy of [B, T, 151936] = ~1.2GB
                # and OOMs on P100. Instead we compute the loss ourselves using F.cross_entropy
                # which handles FP16 inputs without that copy.
                #
                # The shift below is equivalent to what GPT-2 does with x=batch[:,:-1], y=batch[:,1:]
                # but HF models expect the full sequence as input, so we shift the output instead.
                # Given tokens [A,B,C,D,E]: model predicts at each position, then we align
                # predictions [A→?,B→?,C→?,D→?] with targets [B,C,D,E]. Same next-token loss.
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                shift_logits = logits[:, :-1, :].contiguous().view(-1, logits.size(-1))
                shift_labels = labels[:, 1:].contiguous().view(-1)
                loss_full = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
                loss = loss_full / GA
            _sync()
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
        _sync()
        t2 = time.perf_counter()
        micro_dt = t2 - t0

        # micro accounting
        n_tokens = (labels != -100).sum().item()
        tok = max(1, n_tokens)
        tokens += tok
        w_tokens += tok
        w_loss_sum += float(loss_full.item()) * tok
        w_token_count += tok

        micro_count += 1
        micros_in_step += 1

        micro_time += micro_dt
        w_micro_time += micro_dt
        micro_time_min = min(micro_time_min, micro_dt)
        micro_time_max = max(micro_time_max, micro_dt)

        # ----- finish GA -> optimizer -----
        if is_last_micro:
            _sync()
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            sched.step()
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

                step_time_win = w_step_time / max(1, w_step_count)
                micro_time_win = w_micro_time / max(1, w_step_count * GA)
                tok_per_s = w_tokens / elapsed
                lr = optimizer.param_groups[0]["lr"]

                print(f"[{now()}][Epoch {epoch:03d}][Step {step_count:04d}/{steps_per_epoch}] global_step={global_step} "
                      f"train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} lr={lr:.6f} "
                      f"step_time={step_time_win:.3f}s step_time_min={step_time_min:.3f} step_time_max={step_time_max:.3f} "
                      f"micro_time={micro_time_win:.3f}s tp={tok_per_s:.0f} tok/s", flush=True)

                if log is not None:
                    ep = log.setdefault("updates", {}).setdefault(str(epoch), {})
                    ep[f"{step_count:04d}"] = {
                        "global_step": int(global_step),
                        "train_loss":  float(train_loss),
                        "train_ppl":   float(train_ppl),
                        "lr":          float(lr),
                        "step_time":       float(step_time_win),
                        "micro_step_time": float(micro_time_win),
                        "throughput":      float(tok_per_s),
                        "tokens":          int(w_tokens),
                    }

                # reset window
                w_start = time.perf_counter()
                w_step_time = 0.0
                w_step_count = 0
                w_micro_time = 0.0
                w_tokens = 0
                w_loss_sum = 0.0
                w_token_count = 0

            # ----- mini validation -----
            steps_remaining = steps_per_epoch - step_count
            if (val_loader and args.mini_val_every_opt_steps
                    and step_count % args.mini_val_every_opt_steps == 0
                    and steps_remaining >= args.mini_val_every_opt_steps):
                val_metrics = validate(model, val_loader, device, args, max_batches=args.mini_val_max_batches)
                model.train()
                print(f"[{now()}][MiniVal][Epoch {epoch:03d}][Step {step_count:04d}] "
                      f"val_loss={val_metrics['loss']:.4f} val_ppl={val_metrics['ppl']:.2f} val_time={val_metrics['time']:.2f}s",
                      flush=True)
                if log is not None:
                    epm = log.setdefault("minival", {}).setdefault(str(epoch), {})
                    epm[f"{step_count:04d}"] = {
                        "global_step":    int(global_step),
                        "mini_val_loss":  float(val_metrics['loss']),
                        "mini_val_ppl":   float(val_metrics['ppl']),
                        "max_batches":    int(args.mini_val_max_batches),
                    }

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
        "lr":   float(optimizer.param_groups[0]["lr"]),

        "micro_steps": int(micro_count),
        "steps":       int(step_count),
        "tokens":      int(tokens),

        "micro_step_time":     float(micro_time / max(1, micro_count)),
        "micro_step_time_min": float(micro_time_min if micro_count > 0 else float("nan")),
        "micro_step_time_max": float(micro_time_max if micro_count > 0 else float("nan")),
        "step_time":           float(step_time / max(1, step_count)),
        "step_time_min":       float(step_time_min if step_count > 0 else float("nan")),
        "step_time_max":       float(step_time_max if step_count > 0 else float("nan")),
        "epoch_time":          float(time_epoch_s),

        "throughput": float(tok_per_s),
    }


# ------------------------- training driver -------------------------
def train(args, straggle, best_model_group):
    device = torch.device(args.device)

    # ---- ensure cache dirs ----
    data_root = Path(args.data).resolve()
    os.makedirs(data_root, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", str(data_root / ".hf_datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(data_root / ".hf_transformers"))
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

    # ---- tokenizer ----
    print(f"[{now()}] Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, cache_dir=os.environ["TRANSFORMERS_CACHE"],
        trust_remote_code=args.trust_remote_code,
        force_download=args.force_download)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # ---- dataset ----
    print(f"[{now()}] Loading dataset: {args.dataset}")
    raw = load_dataset(args.dataset, cache_dir=os.environ["HF_DATASETS_CACHE"],
                       trust_remote_code=args.trust_remote_code,
                       download_mode=DownloadMode.FORCE_REDOWNLOAD if args.force_download else DownloadMode.REUSE_DATASET_IF_EXISTS)

    # Alpaca has only a 'train' split; carve out a validation set
    if "validation" not in raw:
        print(f"[{now()}] No validation split found; splitting train into train/val ({args.val_split_pct}%)")
        split = raw["train"].train_test_split(test_size=args.val_split_pct / 100.0, seed=args.seed)
        train_examples = split["train"]
        val_examples   = split["test"]
    else:
        train_examples = raw["train"]
        val_examples   = raw["validation"]

    print(f"[{now()}] Tokenizing {len(train_examples)} train + {len(val_examples)} val examples (max_seq_len={args.seq_len})")
    train_ds = AlpacaSFTDataset(train_examples, tokenizer, max_seq_len=args.seq_len, mask_prompt=args.mask_prompt)
    val_ds   = AlpacaSFTDataset(val_examples,   tokenizer, max_seq_len=args.seq_len, mask_prompt=False)

    # ---- samplers + loaders ----
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_ds, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_ds, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=(args.workers > 0),
                              prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
                              collate_fn=_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                            num_workers=args.workers, pin_memory=True,
                            collate_fn=_collate_fn)

    # ---- model ----
    print(f"[{now()}] Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, cache_dir=os.environ["TRANSFORMERS_CACHE"],
        trust_remote_code=args.trust_remote_code,
        force_download=args.force_download,
        torch_dtype=torch.float32,  # P100 needs FP32 master weights; AMP handles FP16 forward
    ).to(device)
    model.config.use_cache = False  # disable KV cache during training
    model.gradient_checkpointing_enable()

    # ---- DDP ----
    model = DDP(model, device_ids=[args.local_rank] if device.type == "cuda" else None,
                broadcast_buffers=False, bucket_cap_mb=args.bucket_cap_mb,
                gradient_as_bucket_view=True, find_unused_parameters=False, static_graph=False)
    model.require_forward_param_sync = False

    # ---- straggle sim ----
    if straggle is not None:
        if straggle.attach(model):
            print(f"[{now()}] {straggle} created and active for rank {args.rank}")
        else:
            print(f"[{now()}] {straggle} inactive for rank {args.rank}")
    else:
        print(f"[{now()}] straggle-sim off")

    # ---- DPA wrapper ----
    if args.backend.startswith("dpa") and dpa is not None:
        model = dpa.DDPWrapper(model, sa_world=args.dpa_k if args.dpa_k else args.world_size,
                               sa_preemptive=args.dpa_preemptive, prescale=args.dpa_prescale)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[{now()}] Dataset: {args.dataset} at {data_root}")
    print(f"  Model: {args.model_name} | Params: {n_params:,} | Vocab: {vocab_size:,} | Seq len: {args.seq_len}")
    print(f"  Train/Val samples: {len(train_ds):,}/{len(val_ds):,}")
    print(f"  LR: {args.learning_rate} | Batch/GPU: {args.batch_size} | GA: {args.gradient_accumulation_steps}")
    print("=" * 60, flush=True)

    # ---- optimizer ----
    decay_params, nodecay_params = [], []
    base = model.module if hasattr(model, "module") else model
    if hasattr(base, "module"): base = base.module  # unwrap DDPWrapper
    for n, p in base.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in ["bias", "LayerNorm", "layer_norm", "layernorm", "rmsnorm", "ln_"]):
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = AdamW([
        {"params": decay_params,   "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8)

    # ---- schedule ----
    GA = max(1, args.gradient_accumulation_steps)
    steps_per_epoch = max(1, len(train_loader) // GA)
    total_steps_planned = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_steps if args.warmup_steps >= 0 else min(100, max(1, total_steps_planned // 10))

    if args.sched == "cosine":
        sched = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps_planned)
    else:
        sched = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps_planned)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

    print(f"[{now()}] Plan: epochs={args.epochs}, batches/epoch={len(train_loader)} "
          f"(GA={GA} -> steps/epoch={steps_per_epoch}), total_steps_planned={total_steps_planned}")
    print(f"[{now()}] LR schedule: {args.sched}, warmup {warmup_steps} steps, min_lr via scheduler")

    if args.log_every_opt_steps > 0:
        if steps_per_epoch >= args.log_every_opt_steps:
            n_logs = steps_per_epoch // args.log_every_opt_steps
            print(f"[{now()}] Periodic logging: every {args.log_every_opt_steps} steps (~{n_logs} times per epoch)")
        else:
            print(f"[{now()}] Periodic logging: disabled (epoch too short: {steps_per_epoch} < {args.log_every_opt_steps})")
    else:
        print(f"[{now()}] Periodic logging: disabled")

    # ---- init JSON log ----
    cfg_json = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    log = {
        "time": now(),
        "data_root": str(data_root),
        "config": cfg_json,
        "plan": {
            "ga": GA,
            "steps_per_epoch":     int(steps_per_epoch),
            "total_steps_planned": int(total_steps_planned),
            "warmup_steps":        int(warmup_steps),
        },
        "vocab_size": vocab_size,
        "n_params": n_params,
        "epochs": {},
        "updates": {},
        "minival": {},
    }
    save_log(args.json, log)
    print(f"[{now()}] Logging to {args.json}")

    best_ppl = float("inf")
    global_step = 0

    dist.barrier()  # make sure all ranks start together
    for epoch in range(args.epochs):
        print(f"[{now()}][Epoch {epoch:03d}] ...")

        epoch_start = time.time()

        if straggle is not None:
            straggle.reset_stats()

        train_loader.sampler.set_epoch(epoch)

        # train
        train_metrics = train_one_epoch(model, train_loader, optimizer, sched, device, scaler, args,
                                        epoch, global_step, val_loader=val_loader, log=log)
        global_step += train_metrics["steps"]

        # validate full
        val_metrics = validate(model, val_loader, device, args, max_batches=args.val_max_batches)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        print(
            f"[{now()}][Epoch {epoch:03d}] "
            f"global_step={global_step} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_ppl={train_metrics['ppl']:.2f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_ppl={val_metrics['ppl']:.2f} "
            f"micro_steps={train_metrics['micro_steps']} "
            f"micro_time={train_metrics['micro_step_time']:.3f}s "
            f"steps={train_metrics['steps']} "
            f"step_time={train_metrics['step_time']:.3f}s "
            f"epoch_train_time={train_metrics['epoch_time']:.3f}s "
            f"epoch_time={epoch_time:.3f}s "
            f"tp={train_metrics['throughput']:.0f} tok/s "
            f"straggle_events={straggle.get_stats()['num_straggle_events'] if straggle and straggle.active else 0}",
            flush=True)

        # JSON epoch log
        epoch_metrics = {
            "train_loss": float(train_metrics["loss"]),
            "train_ppl":  float(train_metrics["ppl"]),
            "val_loss":   float(val_metrics["loss"]),
            "val_ppl":    float(val_metrics["ppl"]),
            "lr":         float(current_lr),

            "micro_steps": int(train_metrics["micro_steps"]),
            "steps":       int(train_metrics["steps"]),
            "global_step": int(global_step),
            "tokens":      int(train_metrics["tokens"]),

            "micro_step_time":     float(train_metrics["micro_step_time"]),
            "micro_step_time_min": float(train_metrics["micro_step_time_min"]),
            "micro_step_time_max": float(train_metrics["micro_step_time_max"]),
            "step_time":           float(train_metrics["step_time"]),
            "step_time_min":       float(train_metrics["step_time_min"]),
            "step_time_max":       float(train_metrics["step_time_max"]),
            "epoch_time":          float(epoch_time),
            "epoch_train_time":    float(train_metrics["epoch_time"]),
            "epoch_train_throughput": float(train_metrics["throughput"]),

            "straggle": straggle.get_stats() if straggle and straggle.active else {},
        }
        log["epochs"][str(epoch)] = epoch_metrics
        save_log(args.json, log)

        if val_metrics["ppl"] < best_ppl:
            best_ppl = val_metrics["ppl"]

    print(f"\n[{now()}] Training complete. Best (local) validation perplexity: {best_ppl:.2f}")

    if args.best_model and best_model_group is not None and args.rank not in args.best_model_ignore:
        best_val_ppl = min(e["val_ppl"] for e in log["epochs"].values())
        t = torch.tensor([best_val_ppl], dtype=torch.float32)
        all_ppls = [torch.zeros(1, dtype=torch.float32) for _ in range(len(args.best_model_active_ranks))]
        dist.all_gather(all_ppls, t, group=best_model_group)
        best_idx  = int(torch.stack(all_ppls).argmin().item())
        best_rank = args.best_model_active_ranks[best_idx]
        pairs = [f"{rank}:{ppl.item():.4f}" for rank, ppl in zip(args.best_model_active_ranks, all_ppls)]
        print(f"[{now()}] All val_ppls: {', '.join(pairs)}")
        print(f"[{now()}] Best val_ppl: {float(all_ppls[best_idx]):.4f} at rank {best_rank}", flush=True)

    if args.save_model:
        model.module.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"[{now()}] Model saved to {args.save_model}", flush=True)

# ------------------------- DDP setup/teardown -------------------------
def setup_ddp(args):
    def env_int(k, d): return d if os.environ.get(k) in (None, "") else int(os.environ.get(k))
    def env_str(k, d): return d if os.environ.get(k) in (None, "") else os.environ.get(k)

    args.rank        = env_int("RANK", args.rank)
    args.world_size  = env_int("WORLD_SIZE", args.world_size)
    args.master_addr = env_str("MASTER_ADDR", args.master_addr)
    args.master_port = env_int("MASTER_PORT", args.master_port)
    args.iface       = env_str("IFACE", args.iface)
    args.local_rank  = (args.rank % torch.cuda.device_count()) if torch.cuda.device_count() else 0

    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)

    os.environ.setdefault("RANK",        str(args.rank))
    os.environ.setdefault("WORLD_SIZE",  str(args.world_size))
    os.environ.setdefault("MASTER_ADDR", args.master_addr)
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ.setdefault("LOCAL_RANK",  str(args.local_rank))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", args.iface)
    os.environ.setdefault("NCCL_SOCKET_IFNAME", args.iface)

    init_method = "env://"

    if args.backend.startswith("dpa") and dpa is not None:
        if not args.dpa_conf:
            raise RuntimeError(f"--dpa_conf required for backend {args.backend}")
        dpa_device  = dpa.DPADeviceOptions.from_config(args.dpa_conf)
        dpa_backend = dpa.DPADpdkBackendOptions.from_config(args.dpa_conf)
        pg_options  = dpa.ProcessGroupDPADpdkOptions(dpa_device, dpa_backend)
        pg_options.hint_pinned_tensor_size = max(600_000_000, args.bucket_cap_mb * (2 ** 20) * 4 if args.bucket_cap_mb is not None else 0)
        pg_options.hint_pinned_tensor_pool_size = 55
        dist.init_process_group(backend=args.backend, init_method=init_method, rank=args.rank, world_size=args.world_size,
                                timeout=datetime.timedelta(seconds=60), pg_options=pg_options)
        if args.dpa_repin:
            os.sched_setaffinity(0, set(range(os.cpu_count() - dpa_backend.threads - 1)))
            print(f"[{now()}] re-pinned to cores 0-{os.cpu_count() - dpa_backend.threads - 1}")
    else:
        dist.init_process_group(backend=args.backend, init_method="env://", rank=args.rank, world_size=args.world_size,
                                timeout=datetime.timedelta(seconds=60))

    print(f"[{now()}] DDP setup with backend={args.backend} world_size={args.world_size} "
          f"master={args.master_addr}:{args.master_port} iface={args.iface} local_rank={args.local_rank}", flush=True)

    args.best_model_group        = None
    args.best_model_active_ranks = None
    best_model_group = None
    if args.best_model:
        args.best_model_active_ranks = [r for r in range(args.world_size) if r not in args.best_model_ignore]
        best_model_group = dist.new_group(ranks=args.best_model_active_ranks, backend="gloo")
        print(f"[{now()}] DDP best model selection ENABLED. Ranks considered: {args.best_model_active_ranks}")
    return best_model_group


# ------------------------- main -------------------------
def main():
    parser = argparse.ArgumentParser(description="Qwen/LLM instruction fine-tuning on Alpaca")

    # DDP/System
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=6)
    parser.add_argument("--iface", type=str, default="ens4f0")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument("--backend", type=str, default="gloo", choices=["nccl", "gloo", "dpa_dpdk"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json", type=str, default="qwen_alpaca.json", help="Path to JSON run log")
    parser.add_argument("--log_every_opt_steps", type=int, default=0, help="Log every N optimizer updates during training. 0=disabled.")

    parser.add_argument("--dpa_conf", type=str, default=None, help="Path to dpa config.json")
    parser.add_argument("--dpa_repin", action="store_true")
    parser.add_argument("--dpa_k", type=int, default=0, help="Configure fastest-k amount. Disabled if 0 or world_size")
    parser.add_argument("--dpa_preemptive", action="store_true", help="Preemptive K-sync: do not wait for STO")
    parser.add_argument("--dpa_prescale", action="store_true", help="Enable prescaling")

    # Model / Dataset
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="HuggingFace model name or path")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca", help="HuggingFace dataset name")
    parser.add_argument("--data", type=str, required=True, help="Root dir for HF caches")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--force_download", action="store_true")
    parser.add_argument("--val_split_pct", type=float, default=5.0, help="Percent of train to use as val if no val split exists")
    parser.add_argument("--mask_prompt", action="store_true", default=True, help="Mask prompt tokens from loss (train only on response)")
    parser.add_argument("--no_mask_prompt", dest="mask_prompt", action="store_false")

    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=-1, help="Warmup in optimizer steps. -1=auto (10%% capped at 100).")
    parser.add_argument("--sched", choices=["cosine", "linear"], default="cosine")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--val_max_batches", type=int, default=0, help="Max batches for end-of-epoch val. 0=all.")
    parser.add_argument("--mini_val_every_opt_steps", type=int, default=0, help="Run mini validation every N optimizer steps. 0=off.")
    parser.add_argument("--mini_val_max_batches", type=int, default=64, help="Batches for mini validation.")

    parser.add_argument("--bucket_cap_mb", type=int, default=None, help="DDP bucket capacity")

    parser.add_argument("--seq_len", type=int, default=512, help="Max sequence length for tokenization")

    # Straggle sim
    def csv_ints(s: str) -> list[int]:
        try: return [int(x) for x in re.split(r"\s*,\s*", s) if x] if s else []
        except ValueError: raise argparse.ArgumentTypeError("Expected comma-separated ints")
    parser.add_argument("--straggle_points", type=int, default=0)
    parser.add_argument("--straggle_prob", type=float, default=0)
    parser.add_argument("--straggle_last", type=int, default=0)
    parser.add_argument("--straggle_skip", type=int, default=0)
    parser.add_argument("--straggle_skip_every", type=int, default=0)
    parser.add_argument("--straggle_ranks", type=csv_ints, default=[])
    parser.add_argument("--straggle_amount", type=float, default=0)
    parser.add_argument("--straggle_multiply", type=float, nargs=2, metavar=("lo", "hi"), default=[1.0, 1.0])
    parser.add_argument("--straggle_verbose", action="store_true")

    parser.add_argument("--best_model", action="store_true", help="Select model with best val ppl among participating ranks")
    parser.add_argument("--best_model_ignore", type=csv_ints, default=[], help="Ranks to exclude from --best_model comparison.")
    parser.add_argument("--save_model", type=str, default="", help="Path to save fine-tuned model after training (rank 0 only). Empty = no save.")

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[{now()}][Warning] Ignoring unknown args: {unknown}", flush=True)

    if args.save_model:
        save_path = os.path.join(args.save_model, f"qwen_rank{args.rank}")
        os.makedirs(save_path, exist_ok=True)
        assert os.access(save_path, os.W_OK), f"Cannot write to {save_path}"
        args.save_model = save_path

    args.local_rank = 0
    args.dpa_dpdk = {}
    if args.dpa_conf:
        with open(args.dpa_conf) as f:
            args.dpa_dpdk = json.load(f).get("dpdk", {})

    if args.dpa_k and args.dpa_k < args.world_size:
        print(f"[{now()}] Straggler mitigation ENABLED with dpa_k={args.dpa_k} !!")
    else:
        print(f"[{now()}] Straggler mitigation DISABLED !!")

    # Determinism
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
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

    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        print(f"[{now()}][Info] Using device=cpu because CUDA is not available", flush=True)
    if args.amp and args.device == "cpu":
        args.amp = False
        print(f"[{now()}][Info] Disabling AMP because CUDA is not available", flush=True)

    args.workers = max(args.workers, 0)
    sys.stdout.reconfigure(line_buffering=True)

    best_model_group = setup_ddp(args)
    print(f"[{now()}] Configuration:\n{json.dumps({k: v for k, v in vars(args).items() if not k.startswith('_')}, indent=2)}")

    straggle = None
    if args.straggle_points and dpa is not None:
        straggle = dpa.DDPStraggleSim(
            points=args.straggle_points, prob=args.straggle_prob,
            amount=args.straggle_amount, ranks=args.straggle_ranks,
            skip=args.straggle_skip, skip_every=args.straggle_skip_every,
            last=args.straggle_last, multiplier_range=args.straggle_multiply,
            verbose=args.straggle_verbose)
        straggle.print_pattern()
    elif args.straggle_points and dpa is None:
        print(f"[{now()}][Warning] --straggle_points={args.straggle_points} but dpa module not available. Skipping.", flush=True)

    train(args, straggle, best_model_group)


if __name__ == "__main__":
    main()