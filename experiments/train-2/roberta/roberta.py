#!/usr/bin/env python3
"""
RoBERTa-base SQuAD v1/v2 DDP fine-tuning trainer.
Training logic follows standard HF run_qa.py conventions.
"""

import os, sys, argparse, time, datetime, json, math, random, re, string, collections
from typing import List, Dict
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    import dpa
except Exception:
    dpa = None

from datasets import load_dataset, DownloadMode
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering,
    default_data_collator, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
)


# ------------------------- utilities -------------------------
def now(): return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def save_log(path, log):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2)
    os.replace(tmp, path)

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self):
        self.sum = 0.0; self.count = 0.0; self.avg = 0.0
    def update(self, val, n=1):
        self.sum += float(val) * n; self.count += n
        self.avg = self.sum / max(1.0, self.count)
    def all_reduce(self):
        if dist.is_available() and dist.is_initialized():
            backend = dist.get_backend()
            device = torch.device(f"cuda:{torch.cuda.current_device()}") if backend == dist.Backend.NCCL else torch.device("cpu")
            t = torch.tensor([self.sum, self.count], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            self.sum, self.count = t.cpu().tolist()
            self.avg = self.sum / max(1.0, self.count)


# ------------------------- data prep (SQuAD-style) -------------------------
def _ensure_caches(root: str):
    os.makedirs(root, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(root, ".hf_datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(root, ".hf_transformers"))
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

def _prepare_features(args, raw, tokenizer):
    max_len, doc_stride = args.max_seq_len, args.doc_stride

    def prep_train(ex):
        tok = tokenizer(
            [q.strip() for q in ex["question"]], ex["context"],
            truncation="only_second", max_length=max_len, stride=doc_stride,
            return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length",
        )
        sample_map = tok.pop("overflow_to_sample_mapping")
        offsets = tok.pop("offset_mapping")
        start_positions, end_positions = [], []
        for i, offs in enumerate(offsets):
            input_ids = tok["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            si = sample_map[i]; ans = ex["answers"][si]
            if len(ans["answer_start"]) == 0:
                start_positions.append(cls_index); end_positions.append(cls_index); continue
            start_char = ans["answer_start"][0]; end_char = start_char + len(ans["text"][0])
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
                start_positions.append(s - 1); end_positions.append(e + 1)
        tok["start_positions"] = start_positions; tok["end_positions"] = end_positions
        return tok

    def prep_val(ex):
        tok = tokenizer(
            [q.strip() for q in ex["question"]], ex["context"],
            truncation="only_second", max_length=max_len, stride=doc_stride,
            return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length",
        )
        sample_map = tok.pop("overflow_to_sample_mapping")
        offsets = tok.pop("offset_mapping")
        start_positions, end_positions, example_id, cleaned_offsets = [], [], [], []
        for i, offs in enumerate(offsets):
            input_ids = tok["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            si = sample_map[i]; ans = ex["answers"][si]
            if len(ans["answer_start"]) == 0:
                start_positions.append(cls_index); end_positions.append(cls_index)
            else:
                start_char = ans["answer_start"][0]; end_char = start_char + len(ans["text"][0])
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
                    start_positions.append(s - 1); end_positions.append(e + 1)
            seq_ids = tok.sequence_ids(i)
            cleaned = [(o if seq_ids[k] == 1 else (0, 0)) for k, o in enumerate(offs)]
            cleaned_offsets.append(cleaned)
            example_id.append(ex["id"][si])
        tok["start_positions"] = start_positions; tok["end_positions"] = end_positions
        tok["example_id"] = example_id; tok["offset_mapping"] = cleaned_offsets
        return tok

    train_features = raw["train"].map(prep_train, batched=True, remove_columns=raw["train"].column_names, desc="Tokenize train")
    val_features   = raw["validation"].map(prep_val, batched=True, remove_columns=raw["validation"].column_names, desc="Tokenize val")

    train_cols = ["input_ids", "attention_mask", "start_positions", "end_positions"]
    if "token_type_ids" in train_features.column_names:
        train_cols.insert(2, "token_type_ids")
    train_features.set_format(type="torch", columns=train_cols)
    val_features.set_format(type=None)
    return train_features, val_features

def _collate_val(batch: List[Dict[str, object]]):
    toT = lambda x: torch.tensor(x, dtype=torch.long)
    input_ids      = toT([b["input_ids"] for b in batch])
    attention_mask = toT([b["attention_mask"] for b in batch])
    token_type_ids = toT([b.get("token_type_ids", [0]*len(b["input_ids"])) for b in batch])
    start_positions= toT([b["start_positions"] for b in batch])
    end_positions  = toT([b["end_positions"] for b in batch])
    example_id     = [b["example_id"] for b in batch]
    offset_mapping = [b["offset_mapping"] for b in batch]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
            "start_positions": start_positions, "end_positions": end_positions,
            "example_id": example_id, "offset_mapping": offset_mapping}


# ------------------------- QA metrics -------------------------
def _norm(s: str) -> str:
    def lower(t): return t.lower()
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def remove_punc(t): return "".join(ch for ch in t if ch not in set(string.punctuation))
    def ws(t): return " ".join(t.split())
    return ws(remove_articles(remove_punc(lower(s))))

def _em(pred: str, golds: List[str]) -> float:
    return float(any(_norm(pred) == _norm(g) for g in golds))

def _f1(pred: str, golds: List[str]) -> float:
    def score(a, b):
        at, bt = _norm(a).split(), _norm(b).split()
        common = collections.Counter(at) & collections.Counter(bt)
        ns = sum(common.values())
        if len(at) == 0 and len(bt) == 0: return 1.0
        if ns == 0: return 0.0
        p, r = ns / len(at), ns / len(bt)
        return 2 * p * r / (p + r + 1e-12)
    return max(score(pred, g) for g in golds) if len(golds) > 0 else score(pred, "")


# ------------------------- validate -------------------------
@torch.no_grad()
def validate(model, loader, device, args, max_batches=0):
    """Per-rank validation on this rank's shard. No cross-rank communication. max_batches=0 means all batches."""
    model.eval()
    losses = AverageMeter()
    tokenizer = args._tokenizer; id2ex = args._id2ex
    preds_local = {}
    is_roberta = getattr((model.module if hasattr(model, "module") else model).config, "model_type", "") == "roberta"

    t0 = time.perf_counter()
    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches: break

        inputs = {
            "input_ids":       batch["input_ids"].to(device, non_blocking=True),
            "attention_mask":  batch["attention_mask"].to(device, non_blocking=True),
            "start_positions": batch["start_positions"].to(device, non_blocking=True),
            "end_positions":   batch["end_positions"].to(device, non_blocking=True),
            "return_dict":     True,
        }
        if not is_roberta:
            inputs["token_type_ids"] = batch["token_type_ids"].to(device, non_blocking=True)

        out = model(**inputs)
        losses.update(out.loss.item(), inputs["input_ids"].size(0))
        s_logits = out.start_logits.detach().cpu()
        e_logits = out.end_logits.detach().cpu()

        for i in range(inputs["input_ids"].size(0)):
            offsets = batch["offset_mapping"][i]; eid = batch["example_id"][i]
            s = s_logits[i]; e = e_logits[i]
            kS = min(int(args.n_best_size), s.numel())
            kE = min(int(args.n_best_size), e.numel())
            s_idx = torch.topk(s, k=kS).indices.tolist()
            e_idx = torch.topk(e, k=kE).indices.tolist()
            best_score, best_text = -1e30, ""
            max_span_len = int(args.max_answer_length)
            for si in s_idx:
                for ej in e_idx:
                    if ej < si or (ej - si + 1) > max_span_len: continue
                    if offsets[si] == (0, 0) or offsets[ej] == (0, 0): continue
                    score = float(s[si].item() + e[ej].item())
                    if score > best_score:
                        st_char, en_char = offsets[si][0], offsets[ej][1]
                        best_score = score
                        best_text = id2ex[eid]["context"][st_char:en_char]
            if args.squad_version == "v2":
                cls_ids = (batch["input_ids"][i] == tokenizer.cls_token_id).nonzero(as_tuple=True)[0]
                if cls_ids.numel() > 0:
                    idx = int(cls_ids[0].item())
                    null_score = float(s[idx].item() + e[idx].item())
                    if (null_score - best_score) > float(args.null_score_diff_threshold):
                        best_score, best_text = null_score, ""
            prev = preds_local.get(eid)
            if (prev is None) or (best_score > prev[0]):
                preds_local[eid] = (best_score, best_text)

    val_loss = losses.avg
    em_sum = f1_sum = 0.0
    for eid, (_, pred_text) in preds_local.items():
        golds = id2ex[eid]["answers"]["text"] or [""]
        em_sum += _em(pred_text, golds)
        f1_sum += _f1(pred_text, golds)
    denom = max(1, len(preds_local))
    em = (em_sum / denom) * 100.0
    f1 = (f1_sum / denom) * 100.0
    val_time = time.perf_counter() - t0

    return {"loss": float(val_loss), "em": float(em), "f1": float(f1), "time": float(val_time)}


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
    is_roberta = getattr((model.module if hasattr(model, "module") else model).config, "model_type", "") == "roberta"

    # ----- epoch counters -----
    epoch_t0 = time.perf_counter()
    micro_count = 0
    step_count  = 0
    samples     = 0

    step_time = 0.0
    step_time_min, step_time_max = float("inf"), 0.0

    # ----- window accumulators (reset each log) -----
    w_start = time.perf_counter()
    w_step_time = 0.0
    w_step_count = 0
    w_samples = 0
    w_loss_sum = 0.0
    w_sample_count = 0

    last_train_loss = float("nan")

    optimizer.zero_grad(set_to_none=True)
    micros_in_step = 0
    step_t0 = None

    for batch_idx, batch in enumerate(dataloader, 1):
        # start timing on first micro of an optimizer step
        if micros_in_step == 0:
            _sync()
            step_t0 = time.perf_counter()

        # ----- forward / backward -----
        inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        if is_roberta:
            inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}

        bs = inputs["input_ids"].size(0)
        samples += bs
        w_samples += bs

        is_last_micro = (micros_in_step == GA - 1)
        ctx = nullcontext() if (GA == 1 or is_last_micro) else model.no_sync()

        with ctx:
            if scaler is not None and scaler.is_enabled():
                with torch.amp.autocast("cuda"):
                    out = model(**inputs); loss_full = out.loss
                    loss = loss_full / GA
                scaler.scale(loss).backward()
            else:
                out = model(**inputs); loss_full = out.loss
                loss = loss_full / GA
                loss.backward()

        w_loss_sum += float(loss_full.item()) * bs
        w_sample_count += bs
        micro_count += 1
        micros_in_step += 1

        # ----- optimizer step (every GA micros) -----
        if is_last_micro:
            _sync()
            if scaler is not None and scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer); scaler.update()
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
                train_loss = (w_loss_sum / max(1, w_sample_count)) if w_sample_count > 0 else float("nan")
                last_train_loss = train_loss
                step_time_win = w_step_time / max(1, w_step_count)
                samp_per_s = w_samples / elapsed
                lr = optimizer.param_groups[0]["lr"]

                print(f"[{now()}][Epoch {epoch:03d}][Step {step_count:04d}/{steps_per_epoch}] global_step={global_step} "
                      f"train_loss={train_loss:.4f} lr={lr:.6f} "
                      f"step_time={step_time_win:.3f}s step_time_min={step_time_min:.3f} step_time_max={step_time_max:.3f} "
                      f"tp={samp_per_s:.0f} samples/s", flush=True)

                if log is not None:
                    ep = log.setdefault("updates", {}).setdefault(str(epoch), {})
                    ep[f"{step_count:04d}"] = {
                        "global_step": int(global_step),
                        "train_loss":  float(train_loss),
                        "lr":          float(lr),
                        "step_time":   float(step_time_win),
                        "throughput":  float(samp_per_s),
                        "samples":     int(w_samples),
                    }

                w_start = time.perf_counter()
                w_step_time = 0.0; w_step_count = 0
                w_samples = 0; w_loss_sum = 0.0; w_sample_count = 0

            # ----- mini validation -----
            steps_remaining = steps_per_epoch - step_count
            if (val_loader and args.mini_val_every_opt_steps
                    and step_count % args.mini_val_every_opt_steps == 0
                    and steps_remaining >= args.mini_val_every_opt_steps):
                val_m = validate(model, val_loader, device, args, max_batches=args.mini_val_max_batches)
                model.train()
                print(f"[{now()}][MiniVal][Epoch {epoch:03d}][Step {step_count:04d}] "
                      f"val_loss={val_m['loss']:.4f} val_em={val_m['em']:.2f}% val_f1={val_m['f1']:.2f}% "
                      f"val_time={val_m['time']:.2f}s", flush=True)
                if log is not None:
                    epm = log.setdefault("minival", {}).setdefault(str(epoch), {})
                    epm[f"{step_count:04d}"] = {
                        "global_step":   int(global_step),
                        "mini_val_loss": float(val_m['loss']),
                        "mini_val_em":   float(val_m['em']),
                        "mini_val_f1":   float(val_m['f1']),
                        "max_batches":   int(args.mini_val_max_batches),
                    }

    # ----- epoch summary -----
    _sync()
    time_epoch_s = time.perf_counter() - epoch_t0
    samp_per_s = samples / max(1e-6, time_epoch_s)

    if not math.isfinite(last_train_loss):
        last_train_loss = w_loss_sum / w_sample_count if w_sample_count > 0 else float("nan")

    return {
        "loss":    float(last_train_loss),
        "lr":      float(optimizer.param_groups[0]["lr"]),
        "steps":   int(step_count),
        "samples": int(samples),

        "step_time":     float(step_time / max(1, step_count)),
        "step_time_min": float(step_time_min if step_count > 0 else float("nan")),
        "step_time_max": float(step_time_max if step_count > 0 else float("nan")),
        "epoch_time":    float(time_epoch_s),
        "throughput":    float(samp_per_s),
    }


# ------------------------- training driver -------------------------
def train(args, straggle, best_model_group):
    device = torch.device(args.device)
    _ensure_caches(args.data)

    ds_name = "squad_v2" if args.squad_version == "v2" else "squad_v1"
    print(f"[{now()}] Loading '{ds_name}' under {args.data}")
    raw = load_dataset(ds_name, cache_dir=os.environ["HF_DATASETS_CACHE"],
                       download_mode=DownloadMode.FORCE_REDOWNLOAD if args.force_download else DownloadMode.REUSE_DATASET_IF_EXISTS)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True,
                                        cache_dir=os.environ["TRANSFORMERS_CACHE"],
                                        force_download=args.force_download)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    train_features, val_features = _prepare_features(args, raw, tok)

    args._id2ex = {ex["id"]: {"context": ex["context"], "answers": ex["answers"]} for ex in raw["validation"]}
    args._tokenizer = tok

    # Samplers + loaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_features, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=args.drop_last_train)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_features, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=args.drop_last_val)

    train_loader = DataLoader(train_features, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=(args.workers > 0),
                              prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
                              collate_fn=default_data_collator)
    val_loader = DataLoader(val_features, batch_size=args.batch_size, sampler=val_sampler, shuffle=False,
                            num_workers=args.workers, pin_memory=True,
                            persistent_workers=(args.workers > 0),
                            prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
                            collate_fn=_collate_val)

    # Model
    cfg = AutoConfig.from_pretrained(args.model_name, cache_dir=os.environ["TRANSFORMERS_CACHE"],
                                     force_download=args.force_download)
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name, config=cfg, cache_dir=os.environ["TRANSFORMERS_CACHE"],
        force_download=args.force_download).to(device)

    model = DDP(model, device_ids=[args.local_rank] if device.type == "cuda" else None,
                broadcast_buffers=False, bucket_cap_mb=args.bucket_cap_mb,
                gradient_as_bucket_view=True, find_unused_parameters=False, static_graph=False)
    model.require_forward_param_sync = False

    # Straggle sim
    if straggle is not None:
        if straggle.attach(model):
            print(f"{straggle} created and active for rank {args.rank}")
        else:
            print(f"{straggle} inactive for rank {args.rank}")

    # DPA wrapper
    if args.backend.startswith("dpa") and dpa is not None:
        model = dpa.DDPWrapper(model, sa_world=args.dpa_k if args.dpa_k else args.world_size, sa_preemptive=args.dpa_preemptive,
                               prescale=args.dpa_prescale)

    print(f"\n[{now()}] Dataset: SQuAD {args.squad_version} at {args.data}")
    print(f"  Model: {args.model_name} | Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Train features: {len(train_features):,} | Val features: {len(val_features):,}")
    print(f"  LR: {args.learning_rate} | Batch/GPU: {args.batch_size} | GA: {args.gradient_accumulation_steps}")
    print("=" * 60, flush=True)

    # Optimizer — standard BERT/RoBERTa param grouping
    base = model.module if hasattr(model, "module") else model
    if hasattr(base, "module"): base = base.module  # unwrap DDPWrapper

    decay_params, nodecay_params = [], []
    for n, p in base.named_parameters():
        if not p.requires_grad: continue
        if any(nd in n for nd in ["bias", "LayerNorm.weight"]):
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0}
    ], lr=args.learning_rate, foreach=True)

    # Schedule
    GA = max(1, args.gradient_accumulation_steps)
    steps_per_epoch = max(1, len(train_loader) // GA)
    total_steps_planned = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_steps if args.warmup_steps >= 0 else int(args.warmup_ratio * total_steps_planned)

    if args.sched == "cosine":
        sched = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps_planned)
    else:
        sched = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps_planned)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda" and args.amp))

    print(f"[{now()}] Plan: epochs={args.epochs}, batches/epoch={len(train_loader)} "
          f"(GA={GA} -> steps/epoch={steps_per_epoch}), total_steps_planned={total_steps_planned}")
    print(f"[{now()}] LR schedule: {args.sched}, warmup {warmup_steps} steps")

    if args.log_every_opt_steps > 0:
        if steps_per_epoch >= args.log_every_opt_steps:
            n_logs = steps_per_epoch // args.log_every_opt_steps
            print(f"[{now()}] Periodic logging: every {args.log_every_opt_steps} steps (~{n_logs} times per epoch)")
        else:
            print(f"[{now()}] Periodic logging: disabled (epoch too short: {steps_per_epoch} < {args.log_every_opt_steps})")
    else:
        print(f"[{now()}] Periodic logging: disabled")

    # Init JSON log
    cfg_json = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    log = {
        "time": now(),
        "config": cfg_json,
        "plan": {
            "ga": GA,
            "steps_per_epoch": int(steps_per_epoch),
            "total_steps_planned": int(total_steps_planned),
            "warmup_steps": int(warmup_steps),
        },
        "epochs": {},
        "updates": {},
        "minival": {},
    }
    save_log(args.json, log)
    print(f"[{now()}] Logging to {args.json}")

    best_em = 0.0; best_f1 = 0.0
    global_step = 0

    dist.barrier()

    for epoch in range(args.epochs):
        print(f"[{now()}][Epoch {epoch:03d}] ...", flush=True)
        epoch_start = time.time()

        if straggle is not None:
            straggle.reset_stats()

        train_loader.sampler.set_epoch(epoch)

        train_m = train_one_epoch(model, train_loader, optimizer, sched, device, scaler, args,
                                  epoch, global_step, val_loader=val_loader, log=log)
        global_step += train_m['steps']

        val_m = validate(model, val_loader, device, args, max_batches=args.val_max_batches)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        print(
            f"[{now()}][Epoch {epoch:03d}] "
            f"global_step={global_step} "
            f"train_loss={train_m['loss']:.4f} "
            f"val_loss={val_m['loss']:.4f} val_em={val_m['em']:.2f}% val_f1={val_m['f1']:.2f}% "
            f"steps={train_m['steps']} "
            f"step_time={train_m['step_time']:.3f}s "
            f"epoch_train_time={train_m['epoch_time']:.3f}s "
            f"epoch_time={epoch_time:.3f}s "
            f"tp={train_m['throughput']:.0f} samples/s "
            f"straggle_events={straggle.get_stats()['num_straggle_events'] if straggle and straggle.active else 0}",
            flush=True
        )

        epoch_metrics = {
            "train_loss": float(train_m['loss']),
            "val_loss":   float(val_m['loss']),
            "val_em":     float(val_m['em']),
            "val_f1":     float(val_m['f1']),
            "lr":         float(current_lr),

            "steps":       int(train_m['steps']),
            "global_step": int(global_step),
            "samples":     int(train_m['samples']),

            "step_time":     float(train_m['step_time']),
            "step_time_min": float(train_m['step_time_min']),
            "step_time_max": float(train_m['step_time_max']),
            "epoch_time":          float(epoch_time),
            "epoch_train_time":    float(train_m['epoch_time']),
            "epoch_train_throughput": float(train_m['throughput']),

            "straggle": straggle.get_stats() if straggle and straggle.active else {}
        }
        log["epochs"][str(epoch)] = epoch_metrics
        save_log(args.json, log)

        best_em = max(best_em, val_m["em"])
        best_f1 = max(best_f1, val_m["f1"])

    print(f"\n[{now()}] Training complete. Best (local) EM: {best_em:.2f}% F1: {best_f1:.2f}%")

    if args.best_model and best_model_group is not None and args.rank not in args.best_model_ignore:
        best_val_f1 = max(e["val_f1"] for e in log["epochs"].values())
        t = torch.tensor([-best_val_f1], dtype=torch.float32)
        all_vals = [torch.zeros(1, dtype=torch.float32) for _ in range(len(args.best_model_active_ranks))]
        dist.all_gather(all_vals, t, group=best_model_group)
        best_idx = int(torch.stack(all_vals).argmin().item())
        best_rank = args.best_model_active_ranks[best_idx]
        pairs = [f"{rank}:{-v.item():.2f}" for rank, v in zip(args.best_model_active_ranks, all_vals)]
        print(f"[{now()}] All val_f1: {', '.join(pairs)}")
        print(f"[{now()}] Best val_f1: {-float(all_vals[best_idx]):.2f} at rank {best_rank}", flush=True)


# ------------------------- DDP setup -------------------------
def setup_ddp(args):
    def env_int(k, d): return d if os.environ.get(k) in (None, "") else int(os.environ.get(k))
    def env_str(k, d): return d if os.environ.get(k) in (None, "") else os.environ.get(k)

    args.rank        = env_int("RANK", args.rank)
    args.world_size  = env_int("WORLD_SIZE", args.world_size)
    args.master_addr = env_str("MASTER_ADDR", args.master_addr)
    args.master_port = env_int("MASTER_PORT", args.master_port)
    args.iface       = env_str("IFACE", args.iface)
    args.local_rank  = (args.rank % torch.cuda.device_count()) if torch.cuda.device_count() else 0

    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)

    os.environ.setdefault("RANK",        str(args.rank))
    os.environ.setdefault("WORLD_SIZE",  str(args.world_size))
    os.environ.setdefault("MASTER_ADDR", args.master_addr)
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ.setdefault("LOCAL_RANK",  str(args.local_rank))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", args.iface)
    os.environ.setdefault("NCCL_SOCKET_IFNAME", args.iface)

    if args.backend.startswith("dpa"):
        if not args.dpa_conf:
            raise RuntimeError(f"--dpa_conf required for backend {args.backend}")
        dpa_device  = dpa.DPADeviceOptions.from_config(args.dpa_conf)
        dpa_backend = dpa.DPADpdkBackendOptions.from_config(args.dpa_conf)
        pg_options  = dpa.ProcessGroupDPADpdkOptions(dpa_device, dpa_backend)
        pg_options.hint_pinned_tensor_size = max(200_000_000,
            args.bucket_cap_mb * (2 ** 20) * 4 if args.bucket_cap_mb is not None else 0)
        pg_options.hint_pinned_tensor_pool_size = 20
        dist.init_process_group(backend=args.backend, init_method="env://",
                                rank=args.rank, world_size=args.world_size,
                                timeout=datetime.timedelta(seconds=60), pg_options=pg_options)
        if args.dpa_repin:
            os.sched_setaffinity(0, set(range(os.cpu_count() - dpa_backend.threads - 1)))
            print(f"[{now()}] re-pinned to cores 0-{os.cpu_count() - dpa_backend.threads - 1}")
    else:
        dist.init_process_group(backend=args.backend, init_method="env://",
                                rank=args.rank, world_size=args.world_size,
                                timeout=datetime.timedelta(seconds=60))

    print(f"[{now()}] DDP setup with backend={args.backend} world_size={args.world_size} "
          f"master={args.master_addr}:{args.master_port} iface={args.iface} local_rank={args.local_rank}", flush=True)

    args.best_model_group = None
    args.best_model_active_ranks = None
    if args.best_model:
        args.best_model_active_ranks = [r for r in range(args.world_size) if r not in args.best_model_ignore]
        best_model_group = dist.new_group(ranks=args.best_model_active_ranks, backend="gloo")
        print(f"[{now()}] DDP best model selection ENABLED. Ranks considered: {args.best_model_active_ranks}")
        return best_model_group


# ------------------------- main -------------------------
def main():
    parser = argparse.ArgumentParser(description='RoBERTa SQuAD DDP fine-tuning')

    # DDP / System
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=6)
    parser.add_argument('--iface', type=str, default='ens4f0')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1')
    parser.add_argument('--master_port', type=int, default=29500)
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo', 'dpa_dpdk'])
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--json', type=str, default='roberta_squad.json', help='Path to JSON run log')
    parser.add_argument('--log_every_opt_steps', type=int, default=0, help='Log every N optimizer updates during training. 0=disabled.')

    parser.add_argument('--dpa_conf', type=str, default=None, help='Path to dpa config.json')
    parser.add_argument('--dpa_repin', action='store_true')
    parser.add_argument('--dpa_k', type=int, default=0, help='Configure fastest-k. 0=disabled.')
    parser.add_argument('--dpa_preemptive', action='store_true',help="Preemptive K-sync: do not wait for STO")
    parser.add_argument('--dpa_prescale', action='store_true', help='Enable prescaling')

    # Dataset
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--squad_version', choices=['v1', 'v2'], default='v1')
    parser.add_argument('--force_download', action='store_true')
    parser.add_argument('--model_name', type=str, default='roberta-base')

    # Training — defaults follow standard RoBERTa SQuAD fine-tuning
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.06)
    parser.add_argument('--warmup_steps', type=int, default=-1,
                        help='Warmup in optimizer steps. -1=use warmup_ratio.')
    parser.add_argument('--sched', choices=['cosine', 'linear'], default='linear')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--drop_last_train', action='store_true')
    parser.add_argument('--drop_last_val', action='store_true')
    parser.add_argument('--val_max_batches', type=int, default=0, help='Max batches for end-of-epoch val. 0=all.')
    parser.add_argument('--mini_val_every_opt_steps', type=int, default=0,
                        help='Run mini validation every N optimizer steps. 0=off.')
    parser.add_argument('--mini_val_max_batches', type=int, default=64,
                        help='Batches to use for mini validation.')

    parser.add_argument('--prescale', action='store_true', help='Prescale gradients for allreduce')
    parser.add_argument('--bucket_cap_mb', type=int, default=None, help='DDP bucket capacity')

    # QA tokenization
    parser.add_argument('--max_seq_len', type=int, default=384)
    parser.add_argument('--doc_stride', type=int, default=128)
    parser.add_argument('--max_answer_length', type=int, default=30)
    parser.add_argument('--n_best_size', type=int, default=20)
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0)

    # Straggle sim
    def csv_ints(s: str) -> list[int]:
        try: return [int(x) for x in re.split(r"\s*,\s*", s) if x] if s else []
        except ValueError: raise argparse.ArgumentTypeError("Expected comma-separated ints")
    parser.add_argument('--straggle_points', type=int, default=0)
    parser.add_argument('--straggle_prob', type=float, default=0)
    parser.add_argument('--straggle_last', type=int, default=0)
    parser.add_argument('--straggle_skip', type=int, default=0)
    parser.add_argument('--straggle_skip_every', type=int, default=0)
    parser.add_argument('--straggle_ranks', type=csv_ints, default=[])
    parser.add_argument('--straggle_amount', type=float, default=0)
    parser.add_argument('--straggle_multiply', type=float, nargs=2, metavar=('lo', 'hi'), default=[1.0, 1.0])
    parser.add_argument('--straggle_verbose', action='store_true')

    parser.add_argument('--best_model', action='store_true',
                        help='Select model with best val F1 among participating ranks')
    parser.add_argument('--best_model_ignore', type=csv_ints, default=[],
                        help='Ranks to exclude from --best_model comparison.')

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[{now()}][Warning] Ignoring unknown args: {unknown}", flush=True)

    args.local_rank = 0
    args.dpa_dpdk = {}
    if args.dpa_conf:
        with open(args.dpa_conf) as f:
            args.dpa_dpdk = json.load(f).get("dpdk", {})

    if args.dpa_k and args.dpa_k < args.world_size:
        print(f"[{now()}] Straggler mitigation ENABLED with dpa_k={args.dpa_k}")
    else:
        print(f"[{now()}] Straggler mitigation DISABLED")

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