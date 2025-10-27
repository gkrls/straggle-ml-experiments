#!/usr/bin/env python
import os, sys, json, time, math, argparse, datetime, random, re
from typing import List, Dict
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

# Optional dpa backend
try:
    import dpa
except Exception:
    dpa = None

from datasets import load_dataset, DownloadMode
from torch.utils.data import Subset
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForQuestionAnswering,
    default_data_collator, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
)

# ------------------------- Utils / Logging -------------------------
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self):
        self.sum = 0.0; self.count = 0.0; self.avg = 0.0; self.min = float("inf"); self.max = 0.0
    def update(self, v, n=1):
        v = float(v); self.sum += v*n; self.count += n; self.avg = self.sum / max(1.0, self.count)
        self.min = min(self.min, v); self.max = max(self.max, v)
    def all_reduce(self):
        if dist.is_available() and dist.is_initialized():
            dev = torch.device(f"cuda:{torch.cuda.current_device()}") if dist.get_backend() == dist.Backend.NCCL else torch.device("cpu")
            t = torch.tensor([self.sum, self.count], dtype=torch.float64, device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            self.sum, self.count = t.tolist()
            self.avg = self.sum / max(1.0, self.count)

def now(): return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def save_log(path, log):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

# ------------------------- Data prep (SQuAD-style) -----------------
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
            # cleaned offsets (only keep context positions)
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
    if "token_type_ids" in train_features.column_names:  # for BERT-like
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

# ------------------------- QA metrics ----------------------------
import string, collections
def _norm(s: str) -> str:
    def lower(t): return t.lower()
    def remove_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def remove_punc(t): return "".join(ch for ch in t if ch not in set(string.punctuation))
    def ws(t): return " ".join(t.split())
    return ws(remove_articles(remove_punc(lower(s))))
def _em(pred: str, golds: List[str]) -> float: return float(any(_norm(pred) == _norm(g) for g in golds))
def _f1(pred: str, golds: List[str]) -> float:
    def score(a, b):
        at, bt = _norm(a).split(), _norm(b).split()
        common = collections.Counter(at) & collections.Counter(bt)
        ns = sum(common.values())
        if len(at) == 0 and len(bt) == 0: return 1.0
        if ns == 0: return 0.0
        p, r = ns/len(at), ns/len(bt); return 2*p*r/(p+r+1e-12)
    return max(score(pred, g) for g in golds) if len(golds) > 0 else score(pred, "")

# ------------------------- Validation (DDP-safe) -----------------
@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    losses = AverageMeter()
    tokenizer = args._tokenizer; id2ex = args._id2ex
    preds_local = {}

    for batch in loader:
        # build inputs, but drop token_type_ids for RoBERTa
        inputs = {
            "input_ids": batch["input_ids"].to(device, non_blocking=True),
            "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
            "start_positions": batch["start_positions"].to(device, non_blocking=True),
            "end_positions": batch["end_positions"].to(device, non_blocking=True),
            "return_dict": True,
        }
        base = model.module if hasattr(model, "module") else model
        if getattr(base.config, "model_type", "") != "roberta":
            inputs["token_type_ids"] = batch["token_type_ids"].to(device, non_blocking=True)

        out = model(**inputs); loss = out.loss
        losses.update(loss.item(), inputs["input_ids"].size(0))
        s_logits = out.start_logits.detach().cpu(); e_logits = out.end_logits.detach().cpu()

        for i in range(inputs["input_ids"].size(0)):
            offsets = batch["offset_mapping"][i]; eid = batch["example_id"][i]
            s = s_logits[i]; e = e_logits[i]
            kS = min(int(args.n_best_size), s.numel()); kE = min(int(args.n_best_size), e.numel())
            s_idx = torch.topk(s, k=kS).indices.tolist(); e_idx = torch.topk(e, k=kE).indices.tolist()
            best_score, best_text = -1e30, ""
            max_span_len = int(args.max_answer_length)
            for si in s_idx:
                for ej in e_idx:
                    if ej < si or (ej - si + 1) > max_span_len: continue
                    if offsets[si] == (0, 0) or offsets[ej] == (0, 0): continue
                    score = float(s[si].item() + e[ej].item())
                    if score > best_score:
                        st_char, en_char = offsets[si][0], offsets[ej][1]
                        best_score = score; best_text = id2ex[eid]["context"][st_char:en_char]
            if args.squad_version == "v2":
                cls_ids = (batch["input_ids"][i] == tokenizer.cls_token_id).nonzero(as_tuple=True)[0]
                if cls_ids.numel() > 0:
                    idx = int(cls_ids[0].item()); null_score = float(s[idx].item() + e[idx].item())
                    if (null_score - best_score) > float(args.null_score_diff_threshold):
                        best_score, best_text = null_score, ""
            prev = preds_local.get(eid)
            if (prev is None) or (best_score > prev[0]): preds_local[eid] = (best_score, best_text)

    # reduce loss
    loss_avg = losses.avg
    if dist.is_available() and dist.is_initialized():
        dev = torch.device(f"cuda:{torch.cuda.current_device()}") if dist.get_backend() == dist.Backend.NCCL else torch.device("cpu")
        t = torch.tensor([losses.sum, losses.count], dtype=torch.float64, device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        loss_avg = (t[0] / max(1.0, t[1].item())).item()

    em = f1 = 0.0
    if dist.is_available() and dist.is_initialized():
        world = dist.get_world_size(); gathered = [None] * world
        dist.all_gather_object(gathered, preds_local)
        if dist.get_rank() == 0:
            merged = {}
            for d in gathered:
                if not d: continue
                for eid, (sc, tx) in d.items():
                    if (eid not in merged) or (sc > merged[eid][0]): merged[eid] = (sc, tx)
            em_sum = f1_sum = 0.0
            for eid, (_, pred_text) in merged.items():
                golds = args._id2ex[eid]["answers"]["text"] or [""]
                em_sum += _em(pred_text, golds); f1_sum += _f1(pred_text, golds)
            denom = max(1, len(merged)); em = (em_sum / denom) * 100.0; f1 = (f1_sum / denom) * 100.0
        return {"loss": float(loss_avg), "em": float(em), "f1": float(f1)} if dist.get_rank() == 0 else {"loss": float(loss_avg), "em": 0.0, "f1": 0.0}
    else:
        em_sum = f1_sum = 0.0
        for eid, (_, pred_text) in preds_local.items():
            golds = args._id2ex[eid]["answers"]["text"] or [""]
            em_sum += _em(pred_text, golds); f1_sum += _f1(pred_text, golds)
        denom = max(1, len(preds_local)); em = (em_sum / denom) * 100.0; f1 = (f1_sum / denom) * 100.0
        return {"loss": float(loss_avg), "em": float(em), "f1": float(f1)}


# ------------------------- Train one epoch ----------------------
def train_one_epoch(model, loader, minival_loader, optim, sched, device, scaler, args, epoch):
    model.train()
    losses = AverageMeter(); step_time = AverageMeter(); data_time = AverageMeter()
    if device.type == "cuda":
        e_start = torch.cuda.Event(True); e_end = torch.cuda.Event(True); e_start.record()
    else: e_start = time.perf_counter()
    step_start = time.perf_counter(); samples = 0.0; total_steps = len(loader)

    for step, batch in enumerate(loader, 1):
        cur = time.perf_counter(); data_time.update(cur - step_start)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        # Drop token_type_ids for RoBERTa
        base = model.module if hasattr(model, "module") else model
        if getattr(base.config, "model_type", "") == "roberta":
            batch = {k: v for k, v in batch.items() if k != "token_type_ids"}

        bs = batch["input_ids"].size(0); samples += bs
        optim.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                out = model(**batch); loss = out.loss
            scaler.scale(loss).backward(); scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optim); scaler.update(); sched.step()
        else:
            out = model(**batch); loss = out.loss
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step(); sched.step()
        losses.update(loss.item(), bs)
        elapsed = time.perf_counter() - step_start; step_time.update(elapsed); step_start = time.perf_counter()

        if args.log_interval and args.rank == 0 and (step == 1 or step % args.log_interval == 0 or step == total_steps):
            inst_tp = bs / max(1e-9, elapsed)
            print(f"[{now()}][Epoch {epoch:03d} Step {step:05d}/{total_steps}] "
                  f"loss={loss.item():.4f} avg_loss={losses.avg:.4f} "
                  f"lr={optim.param_groups[0]['lr']:.6f} step_time={elapsed:.3f}s "
                  f"data={data_time.avg:.3f}s comp={max(0.0, step_time.avg - data_time.avg):.3f}s "
                  f"tp=~{inst_tp:.1f} samples/sec", flush=True)
        # if args.val_every and ((step + 1) % int(args.val_every) == 0):
        if args.val_every and (step % int(args.val_every) == 0):
            mid = validate(model, minival_loader, device, args)
            # if args.rank == 0:
            print(
                f"[{now()}][MiniVal][Epoch {epoch:03d} Step {step:05d}] "
                f"val_loss={mid['loss']:.4f} val_em={mid['em']:.2f}% val_f1={mid['f1']:.2f}%",
                flush=True
            )
            if args.json:
                try:
                    global_step = int(epoch * total_steps + step)  # cumulative; use `step` alone if you prefer per-epoch
                    with open(args.json, "r") as f:
                        log = json.load(f)
                        updates_minival = log.setdefault("minival", {})
                        epm = updates_minival.setdefault(str(epoch), {})
                        epm[f"{step:04d}"] = {
                            "global_step": int(global_step),
                            "mini_val_em": float(mid['em']),
                            "mini_val_f1":  float(mid['f1']),
                            "lr": float(optim.param_groups[0]["lr"])
                        }
                    save_log(args.json, log)
                except Exception as e:
                    print(f"[{now()}][Warning] Failed to update JSON log (mini-val): {e}", flush=True)
                  
            # validate() sets model.eval(); switch back to train
            model.train()

    if device.type == "cuda":
        e_end.record(); e_end.synchronize(); dur = e_start.elapsed_time(e_end) / 1000.0
    else: dur = time.perf_counter() - e_start
    tp = samples / max(1e-6, dur)
    local_loss = losses.avg; losses.all_reduce()
    return {"loss_global": float(losses.avg), "loss": float(local_loss),
            "step_time_min": float(step_time.min), "step_time_max": float(step_time.max),
            "step_time": float(step_time.avg), "data_time": float(data_time.avg),
            "comp_time": float(step_time.avg - data_time.avg), "epoch_time": float(dur),
            "throughput": float(tp)}

# ------------------------- Main train ---------------------------
def train(args):
    device = torch.device(args.device); _ensure_caches(args.data)
    name = "squad_v2" if args.squad_version == "v2" else "squad_v1"
    print(f"[Data] Loading '{name}' under {args.data}")
    raw = load_dataset(name, cache_dir=os.environ["HF_DATASETS_CACHE"],
                       download_mode=DownloadMode.FORCE_REDOWNLOAD if args.force_download else DownloadMode.REUSE_DATASET_IF_EXISTS)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir=os.environ["TRANSFORMERS_CACHE"], force_download=args.force_download)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token  # guard for rare edge-cases
    train_features, val_features = _prepare_features(args, raw, tok)

    # id -> example (for metrics)
    args._id2ex = {ex["id"]: {"context": ex["context"], "answers": ex["answers"]} for ex in raw["validation"]}
    args._tokenizer = tok

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_features, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=args.drop_last_train)
    val_sampler   = torch.utils.data.distributed.DistributedSampler(val_features,   num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=args.drop_last_val)
    train_loader = DataLoader(train_features, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers > 0),
                              prefetch_factor=args.prefetch_factor, collate_fn=default_data_collator)
    val_loader   = DataLoader(val_features, batch_size=args.batch_size, sampler=val_sampler, shuffle=False,
                              num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers > 0),
                              prefetch_factor=args.prefetch_factor, collate_fn=_collate_val)

    minival_loader = None
    if args.val_every:
      frac = float(args.val_every_subset) # float(getattr(args, "val_subset", 0.25))
      if not (0.0 < frac <= 1.0):
          frac = 0.25
      if frac < 1.0:
          idx = list(range(len(val_features)))
          rnd = random.Random(123)   # deterministic subset
          rnd.shuffle(idx)
          k = max(1, int(len(idx) * frac))
          sub_idx = idx[:k]
          sub_ds = Subset(val_features, sub_idx)
          sub_sampler = torch.utils.data.distributed.DistributedSampler(
              sub_ds, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=args.drop_last_val)
          minival_loader = DataLoader(sub_ds, batch_size=args.batch_size, sampler=sub_sampler, shuffle=False,
              num_workers=args.workers, pin_memory=True, persistent_workers=(args.workers > 0),
              prefetch_factor=args.prefetch_factor, collate_fn=_collate_val)
      else:
          minival_loader = val_loader
    else:
        minival_loader = val_loader




    cfg = AutoConfig.from_pretrained(args.model_name, cache_dir=os.environ["TRANSFORMERS_CACHE"], force_download=args.force_download)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name, config=cfg, cache_dir=os.environ["TRANSFORMERS_CACHE"], force_download=args.force_download).to(device)

    model = DDP(model, device_ids=[args.local_rank] if device.type == "cuda" else None,
                gradient_as_bucket_view=True, find_unused_parameters=False, static_graph=args.static_graph)

    # Optional dpa wrapper
    if args.backend.startswith("dpa") and dpa is not None:
        model = dpa.DDPWrapper(model, straggle=args.world_size, prescale=args.prescale)
    # Straggle sim
    straggle = dpa.DDPStraggleSim(points=args.straggle_points, prob=args.straggle_prob, amount=args.straggle_amount, ranks=args.straggle_ranks)    
    if straggle.attach(model): print(f"Straggle sim initialized with {straggle}")
    else: print(f"Straggle sim inactive")
 
    # Optim + sched
    base = model.module if hasattr(model, "module") else model
    decay, no_decay = [], []
    for n, p in base.named_parameters():
        (no_decay if any(nd in n for nd in ["bias", "LayerNorm.weight"]) else decay).append(p)
    optim = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                               {"params": no_decay, "weight_decay": 0.0}],
                              lr=args.learning_rate, foreach=True)
    total_steps = len(train_loader) * args.epochs
    # sched = get_linear_schedule_with_warmup(optim, int(args.warmup_ratio * total_steps), total_steps)
    if getattr(args, "sched", "cosine") == "cosine":
      sched = get_cosine_schedule_with_warmup(optim, int(args.warmup_ratio * total_steps), total_steps)
    else:
      sched = get_linear_schedule_with_warmup(optim, int(args.warmup_ratio * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    # JSON header
    cfg_json = {k: v for k, v in vars(args).items() if not k.startswith("_")}
    log = {"time": now(), "config": cfg_json, "epochs": {}}
    save_log(args.json, log)
    print(f"[{now()}] Logging to {args.json}")

    best_em = best_f1 = 0.0
    for epoch in range(args.epochs):
        print(f"[{now()}][Epoch {epoch:03d}] ...")
        epoch_start = time.time()

        straggle.reset_stats()

        train_loader.sampler.set_epoch(epoch)

        train_m = train_one_epoch(model, train_loader, minival_loader, optim, sched, device, scaler, args, epoch)
        val_m   = validate(model, val_loader, device, args)
        epoch_time = time.time() - epoch_start
        cur_lr = optim.param_groups[0]["lr"]

        print(f"[{now()}][Epoch {epoch:03d}] train_loss={train_m['loss']:.4f} (global={train_m['loss_global']:.4f}) "
              f"val_loss={val_m['loss']:.4f} val_em={val_m['em']:.2f}% val_f1={val_m['f1']:.2f}% "
              f"lr={cur_lr:.6f} steps={int(len(train_loader))} epoch_time={epoch_time:.2f}s "
              f"step_time={train_m['step_time']:.2f}s (min={train_m['step_time_min']:.2f}s, max={train_m['step_time_max']:.2f}s) "
              f"tp=~{train_m['throughput']:.1f} samples/s",
              f"straggle_events={straggle.get_stats().get('num_straggle_events', 0)}", flush=True)

        epoch_log = {
            "lr": float(cur_lr),
            "train_loss": float(train_m["loss"]),
            "train_loss_global": float(train_m["loss_global"]),
            "steps": int(len(train_loader)),
            "step_time_min": float(train_m["step_time_min"]),
            "step_time_max": float(train_m["step_time_max"]),
            "step_time": float(train_m["step_time"]),
            "data_time": float(train_m["data_time"]),
            "comp_time": float(train_m["comp_time"]),
            "epoch_time": float(epoch_time),
            "epoch_train_time": float(train_m["epoch_time"]),
            "epoch_train_throughput": float(train_m["throughput"]),
            "val_loss": float(val_m["loss"]),
            "val_em": float(val_m["em"]),
            "val_f1": float(val_m["f1"]),
            "straggle": straggle.get_stats() if straggle.active else {}
        }

        with open(args.json, "r") as f:
            log = json.load(f)
        log["epochs"][str(epoch)] = epoch_log
        save_log(args.json, log)

        best_em = max(best_em, val_m["em"]); best_f1 = max(best_f1, val_m["f1"])


# # ------------------------- DDP setup ----------------------------
# def setup_ddp(args):
#     def env_i(k, d): return d if os.environ.get(k) in (None, "") else int(os.environ.get(k))
#     def env_s(k, d): return d if os.environ.get(k) in (None, "") else os.environ.get(k)
#     args.rank = env_i("RANK", args.rank); args.world_size = env_i("WORLD_SIZE", args.world_size)
#     args.master_addr = env_s("MASTER_ADDR", args.master_addr); args.master_port = env_i("MASTER_PORT", args.master_port)
#     if os.environ.get("LOCAL_RANK") is not None:
#         args.local_rank = int(os.environ["LOCAL_RANK"])
#     elif torch.cuda.device_count(): args.local_rank = (args.rank % torch.cuda.device_count())
#     else: args.local_rank = 0
#     if args.device == "cuda" and torch.cuda.is_available(): torch.cuda.set_device(args.local_rank)
#     os.environ.setdefault("RANK", str(args.rank)); os.environ.setdefault("WORLD_SIZE", str(args.world_size))
#     os.environ.setdefault("MASTER_ADDR", args.master_addr); os.environ.setdefault("MASTER_PORT", str(args.master_port))
#     os.environ.setdefault("LOCAL_RANK", str(args.local_rank))
#     dist.init_process_group(backend=args.backend, init_method="env://", rank=args.rank, world_size=args.world_size, timeout=datetime.timedelta(seconds=60))
#     print(f"[DDP] backend={args.backend} world_size={args.world_size} master={args.master_addr}:{args.master_port} local_rank={args.local_rank}", flush=True)


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

    # Initialize process group
    if args.backend.startswith("dpa"):
        if not args.dpa_conf: raise RuntimeError(f"--dpa_conf required for backend {args.backend}")
        dpa_device = dpa.DPADeviceOptions.from_config(args.dpa_conf)
        dpa_backend = dpa.DPADpdkBackendOptions.from_config(args.dpa_conf)
        pg_options = dpa.ProcessGroupDPADpdkOptions(dpa_device, dpa_backend)
        pg_options.hint_pinned_tensor_size = max(200_000_000, args.bucket_cap_mb * (2 ** 20) * 4 if args.bucket_cap_mb is not None else 0)
        pg_options.hint_pinned_tensor_pool_size = 20                                                                                      
        dist.init_process_group(backend=args.backend, init_method="env://", rank=args.rank, world_size=args.world_size, timeout = datetime.timedelta(seconds=60), pg_options=pg_options)
    else:
        dist.init_process_group(backend=args.backend, init_method="env://", rank=args.rank, world_size=args.world_size, timeout=datetime.timedelta(seconds=60))
    # Start the process group
    # dist.init_process_group(backend=args.backend, init_method="env://", rank=args.rank, world_size=args.world_size, timeout=datetime.timedelta(seconds=30))

    print(f"[DDP] backend={args.backend} world_size={args.world_size} "
          f"master={args.master_addr}:{args.master_port} iface={args.iface} local_rank={args.local_rank}", flush=True)


def main():
    p = argparse.ArgumentParser()
    # DDP/System
    p.add_argument("--rank", type=int, default=0); p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--iface", type=str, default="ens4f0")
    p.add_argument("--master_addr", type=str, default="42.0.0.1"); p.add_argument("--master_port", type=int, default=29500)
    p.add_argument("--backend", type=str, default="nccl", choices=["nccl", "gloo", "dpa_dpdk"])
    p.add_argument("--dpa_conf", type=str, default=None, help="Path to dpa config.json")
    p.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument('--seed', type=int, default=42)
    p.add_argument("--workers", type=int, default=4)

    # Logging
    p.add_argument("--json", type=str, default="roberta_squad.json"); p.add_argument("--log_interval", type=int, default=0)
    p.add_argument("--val_every", type=int, default=0, help="Run mid-epoch eval every N steps (0=off)")
    p.add_argument("--val_every_subset", type=float, default=0.25, help="Fraction of dev set for mid-epoch eval (0<val<=1)")

    # Dataset
    p.add_argument("--squad_version", choices=["v1", "v2"], default="v1")
    p.add_argument("--data", type=str, required=True); p.add_argument("--force_download", action="store_true")

    # Training/model
    p.add_argument("--sched", choices=["cosine","linear"], default="cosine")
    p.add_argument("--model_name", type=str, default="roberta-base")
    p.add_argument("--epochs", type=int, default=3); p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=3e-5); p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--amp", action="store_true"); p.add_argument("--drop_last_train", action="store_true")
    p.add_argument("--drop_last_val", action="store_true"); p.add_argument("--static_graph", action="store_true")
    p.add_argument("--prefetch_factor", type=int, default=2); p.add_argument("--max_grad_norm", type=float, default=1.0)

    # QA tokenization
    p.add_argument("--max_seq_len", type=int, default=384)
    p.add_argument("--doc_stride", type=int, default=128)
    p.add_argument("--max_answer_length", type=int, default=30)
    p.add_argument("--n_best_size", type=int, default=20)
    p.add_argument("--null_score_diff_threshold", type=float, default=0.0)

    p.add_argument('--prescale', action="store_true", help="Prescale gradients for allreduce")
    p.add_argument("--bucket_cap_mb", type=int, default=None, help="DDP bucket capacity")

        # Straggle
    def csv_ints(s: str) -> List[int]:
        if not s: return []
        try: return [int(x) for x in re.split(r"\s*,\s*", s) if x]
        except ValueError: raise argparse.ArgumentTypeError("Expected a comma-separated list of integers (e.g. 1,2,3)")
    p.add_argument("--straggle_points", type=int, help="Number of straggle points (1-3). Use 0 for no straggle sim", default=0)
    p.add_argument("--straggle_prob", type=float, help="Probability to straggle at each point", default=0)
    p.add_argument("--straggle_ranks", type=csv_ints, help="comma separated list of ints", default=[])
    p.add_argument("--straggle_amount", type=float, help="base straggle amount in seconds (e.g. mean step time)", default=0)
    p.add_argument("--straggle_multiply", type=float, nargs=2, metavar=("lo","hi"), help="straggle amount multipler lo and hi", default=[1.0, 1.0])
    p.add_argument("--straggle_verbose", action='store_true')


    args = p.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    args.seed = args.seed + args.rank * 1000
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
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"; print("[Info] CUDA not available, switching to CPU", flush=True)
    if args.amp and args.device == "cpu":
        args.amp = False; print("[Info] Disabling AMP on CPU", flush=True)
    if args.workers < 1: args.workers = 1

    cfg = {k: v for k, v in vars(args).items() if not k.startswith('_')}
    print(json.dumps(cfg, indent=2))

    sys.stdout.reconfigure(line_buffering=True)
    setup_ddp(args)
    try:
        train(args)
    finally:
        if dist.is_available() and dist.is_initialized(): dist.destroy_process_group()

if __name__ == "__main__":
    main()
