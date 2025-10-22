#!/usr/bin/env python3

import argparse
import os
import sys
import json
import datetime
import time
import re
import random
import numpy as np
import torch
import torch.distributed as dist
from typing import List

# DPA module
import dpa

# HuggingFace
from datasets import load_dataset, DownloadMode
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    default_data_collator,
    set_seed,
)
import evaluate

def setup_ddp(args):
    def env_int(key: str, default: int):
        val = os.environ.get(key)
        if val is None: return default
        try: return int(val)
        except ValueError: return default
    
    def env_str(key: str, default: str):
        val = os.environ.get(key)
        return val if val is not None else default
    
    args.rank = env_int("RANK", args.rank)
    args.world_size = env_int("WORLD_SIZE", args.world_size)
    args.master_addr = env_str("MASTER_ADDR", args.master_addr)
    args.master_port = env_int("MASTER_PORT", args.master_port)
    args.iface = env_str("IFACE", args.iface)
    
    env_local_rank = os.environ.get("LOCAL_RANK")
    if env_local_rank is not None:
        args.local_rank = int(env_local_rank)
    elif torch.cuda.device_count():
        args.local_rank = (args.rank % torch.cuda.device_count())
    else:
        args.local_rank = 0
    
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
    
    # Initialize process group
    if args.backend.startswith("dpa"):
        if not args.dpa_conf:
            raise RuntimeError(f"--dpa_conf required for backend {args.backend}")
        
        dpa_device = dpa.DPADeviceOptions.from_config(args.dpa_conf)
        dpa_backend = dpa.DPADpdkBackendOptions.from_config(args.dpa_conf)
        pg_options = dpa.ProcessGroupDPADpdkOptions(dpa_device, dpa_backend)
        pg_options.hint_pinned_tensor_size = max(200_000_000, args.bucket_cap_mb * (2 ** 20) * 4 if args.bucket_cap_mb else 0)
        pg_options.hint_pinned_tensor_pool_size = 20
        dist.init_process_group(
            backend=args.backend, init_method="env://",
            rank=args.rank, world_size=args.world_size,
            timeout=datetime.timedelta(seconds=60), pg_options=pg_options
        )
    else:
        dist.init_process_group(
            backend=args.backend, init_method="env://",
            rank=args.rank, world_size=args.world_size,
            timeout=datetime.timedelta(seconds=60)
        )
    
    print(f"[DDP] backend={args.backend} world_size={args.world_size} "
          f"master={args.master_addr}:{args.master_port} iface={args.iface} local_rank={args.local_rank}", 
          flush=True)


# ========================= OFFICIAL HF DATA PREPROCESSING =========================

def prepare_datasets(args):
    """Load and prepare datasets using official HF approach"""
    
    # Setup data directories
    os.makedirs(args.data, exist_ok=True)
    ds_cache = os.path.join(args.data, ".hf_datasets")
    tf_cache = os.path.join(args.data, ".hf_transformers")
    os.makedirs(ds_cache, exist_ok=True)
    os.makedirs(tf_cache, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = ds_cache
    os.environ["TRANSFORMERS_CACHE"] = tf_cache
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, cache_dir=tf_cache
    )
    
    # Load dataset
    dataset_name = "squad" if args.squad_version == "v1" else "squad_v2"
    dl_mode = DownloadMode.FORCE_REDOWNLOAD if args.force_download else DownloadMode.REUSE_DATASET_IF_EXISTS
    datasets = load_dataset(dataset_name, cache_dir=ds_cache, download_mode=dl_mode)
    
    # Preprocessing functions (official HF approach)
    pad_on_right = tokenizer.padding_side == "right"
    max_length = args.max_seq_len
    doc_stride = args.doc_stride
    
    def prepare_train_features(examples):
        examples["question"] = [q.lstrip() for q in examples["question"]]
        
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")
        
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            
            sequence_ids = tokenized_examples.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1
                
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1
                
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
        
        return tokenized_examples
    
    def prepare_validation_features(examples):
        examples["question"] = [q.lstrip() for q in examples["question"]]
        
        tokenized_examples = tokenizer(
            examples["question" if pad_on_right else "context"],
            examples["context" if pad_on_right else "question"],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        tokenized_examples["example_id"] = []
        
        for i in range(len(tokenized_examples["input_ids"])):
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        
        return tokenized_examples
    
    # Process datasets
    train_dataset = datasets["train"].map(
        prepare_train_features,
        batched=True,
        remove_columns=datasets["train"].column_names,
    )
    
    eval_dataset = datasets["validation"].map(
        prepare_validation_features,
        batched=True,
        remove_columns=datasets["validation"].column_names,
    )
    
    return train_dataset, eval_dataset, datasets["validation"], tokenizer


def postprocess_qa_predictions(examples, features, predictions, n_best_size=20, max_answer_length=30):
    """Simplified postprocessing for evaluation"""
    import collections
    
    all_start_logits, all_end_logits = predictions
    
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)
    
    predictions = collections.OrderedDict()
    
    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        
        valid_answers = []
        context = example["context"]
        
        for feature_index in feature_indices:
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = features[feature_index]["offset_mapping"]
            
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            
            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append({
                        "score": start_logits[start_index] + end_logits[end_index],
                        "text": context[start_char: end_char]
                    })
        
        if valid_answers:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            best_answer = {"text": "", "score": 0.0}
        
        predictions[example["id"]] = best_answer["text"]
    
    return predictions


# ========================= CUSTOM LOGGING CALLBACK =========================

class CustomLoggingCallback(TrainerCallback):
    def __init__(self, args, straggle_sim=None):
        self.args = args
        self.straggle_sim = straggle_sim
        self.epoch_start_time = None
        self.current_epoch = -1
        self.train_losses = []
        self.log_data = {
            "time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "config": {k: v for k, v in vars(args).items() if not k.startswith('_')},
            "epochs": {}
        }
        self.save_json_log()
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of an epoch"""
        self.current_epoch = int(state.epoch) if state.epoch else 0
        self.epoch_start_time = time.time()
        self.train_losses = []
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{now}][Epoch {self.current_epoch:03d}] ...", flush=True)
        
        # Reset straggle stats if available
        if self.straggle_sim and hasattr(self.straggle_sim, 'reset_stats'):
            self.straggle_sim.reset_stats()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging (every logging_steps)"""
        if logs and "loss" in logs:
            self.train_losses.append(logs["loss"])
            
        # Log at intervals if requested
        if self.args.log_interval > 0 and state.global_step % self.args.log_interval == 0:
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            current_lr = logs.get("learning_rate", 0)
            loss = logs.get("loss", 0)
            print(
                f"[{now}][Epoch {self.current_epoch:03d} Step {state.global_step:05d}] "
                f"loss={loss:.4f} lr={current_lr:.6f}",
                flush=True
            )
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called after evaluation"""
        if metrics and self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            
            # Calculate average training loss
            train_loss = sum(self.train_losses) / len(self.train_losses) if self.train_losses else 0
            
            # Get metrics
            val_loss = metrics.get("eval_loss", 0)
            val_em = metrics.get("eval_exact_match", metrics.get("eval_exact", 0))
            val_f1 = metrics.get("eval_f1", 0)
            
            # Get learning rate
            lr = 0
            if hasattr(state, 'log_history'):
                for log_entry in reversed(state.log_history):
                    if 'learning_rate' in log_entry:
                        lr = log_entry['learning_rate']
                        break
            
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            straggle_events = 0
            if self.straggle_sim and hasattr(self.straggle_sim, 'get_stats'):
                straggle_events = self.straggle_sim.get_stats().get('num_straggle_events', 0)
            
            print(
                f"[{now}][Epoch {self.current_epoch:03d}] "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_em={val_em:.2f}% val_f1={val_f1:.2f}% "
                f"lr={lr:.6f} steps={state.global_step} epoch_time={epoch_time:.2f}s "
                f"straggle_events={straggle_events}",
                flush=True
            )
            
            # Save to JSON log
            epoch_data = {
                "lr": float(lr),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_em": float(val_em),
                "val_f1": float(val_f1),
                "steps": int(state.global_step),
                "epoch_time": float(epoch_time),
                "straggle": self.straggle_sim.get_stats() if self.straggle_sim and hasattr(self.straggle_sim, 'get_stats') else {}
            }
            
            self.log_data["epochs"][str(self.current_epoch)] = epoch_data
            self.save_json_log()
    
    def save_json_log(self):
        if self.args.rank in [0, -1]:  # Only main process
            tmp = f"{self.args.json}.tmp"
            with open(tmp, "w") as f:
                json.dump(self.log_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.args.json)


# ========================= MAIN TRAINING FUNCTION =========================

def train(args):
    """Main training using official HF Trainer"""
    device = torch.device(args.device)
    
    # Load datasets
    train_dataset, eval_dataset, eval_examples, tokenizer = prepare_datasets(args)
    
    # Load model
    config = AutoConfig.from_pretrained(
        args.model_name,
        cache_dir=os.path.join(args.data, ".hf_transformers")
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        args.model_name,
        config=config,
        cache_dir=os.path.join(args.data, ".hf_transformers")
    ).to(device)
    
    # Setup metrics
    metric = evaluate.load("squad" if args.squad_version == "v1" else "squad_v2")
    
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        start_logits, end_logits = predictions
        
        # Postprocess
        predictions = postprocess_qa_predictions(
            eval_examples,
            eval_dataset,
            (start_logits, end_logits),
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
        )
        
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        references = [
            {"id": ex["id"], "answers": ex["answers"]} for ex in eval_examples
        ]
        
        results = metric.compute(predictions=formatted_predictions, references=references)
        return {
            "exact_match": results.get("exact_match", results.get("exact", 0)),
            "f1": results.get("f1", 0)
        }
    
    # Setup straggle simulation
    straggle_sim = dpa.DDPStraggleSim(
        points=args.straggle_points,
        prob=args.straggle_prob,
        amount=args.straggle_amount,
        ranks=args.straggle_ranks
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.data, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=max(1, args.log_interval) if args.log_interval > 0 else 500,
        fp16=args.amp,
        gradient_accumulation_steps=1,
        dataloader_num_workers=args.workers,
        dataloader_drop_last=args.drop_last_train,
        max_grad_norm=args.max_grad_norm,
        report_to=[],  # Disable default reporting
        disable_tqdm=False,
        local_rank=args.local_rank if args.world_size > 1 else -1,
        ddp_backend="nccl" if args.backend == "nccl" else "gloo",
        label_names=["start_positions", "end_positions"],
    )
    
    # Create custom logging callback
    logging_callback = CustomLoggingCallback(args, straggle_sim)
    
    # Create trainer with official HF logic
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[logging_callback],  # Add our custom logging
    )
    
    # NOW WRAP THE TRAINER'S MODEL WITH DPA
    if args.backend.startswith("dpa"):
        print(f"[Rank {args.rank}] Wrapping model with DPA DDPWrapper...")
        trainer.model = dpa.DDPWrapper(trainer.model, straggle=args.world_size, prescale=args.prescale)
        
    # Attach straggle sim
    if straggle_sim.attach(trainer.model):
        print(f"Straggle sim initialized with {straggle_sim}")
    else:
        print(f"Straggle sim inactive")
    
    # Train!
    print(f"[Rank {args.rank}] Starting training...")
    trainer.train()
    
    # Final evaluation
    if args.rank == 0:
        print("\n===== Training Complete =====")
        final_metrics = trainer.evaluate()
        print(f"Final Results:")
        print(f"  EM: {final_metrics.get('eval_exact_match', 0):.2f}%")
        print(f"  F1: {final_metrics.get('eval_f1', 0):.2f}%")


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--iface', type=str, default="ens4f0")
    parser.add_argument('--master_addr', type=str, default="42.0.0.1")
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument('--backend', type=str, default='gloo', choices=['nccl', 'gloo', 'dpa_dpdk'])
    parser.add_argument("--dpa_conf", type=str, default=None, help="Path to dpa config.json")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument("--deterministic", action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    
    # Logging
    parser.add_argument("--json", type=str, default="bert_squad.json", help="Path to JSON run log")
    parser.add_argument('--log_interval', type=int, default=0, help="Steps between progress prints")
    
    # Dataset
    parser.add_argument('--squad_version', type=str, choices=['v1','v2'], default='v1')
    parser.add_argument('--data', type=str, required=True, help='Single directory (created if missing). Everything stays under it.')
    parser.add_argument('--force_download', action='store_true', help='Force re-download')
    
    # Training/model
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help="HF model for QA")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--drop_last_train", action='store_true')
    parser.add_argument("--drop_last_val", action='store_true')
    parser.add_argument("--static_graph", action='store_true')
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    
    parser.add_argument('--prescale', action="store_true", help="Prescale gradients")
    parser.add_argument("--bucket_cap_mb", type=int, default=None, help="DDP bucket capacity")
    
    # QA tokenization
    parser.add_argument('--max_seq_len', type=int, default=384)
    parser.add_argument('--doc_stride', type=int, default=128)
    parser.add_argument('--max_answer_length', type=int, default=30)
    parser.add_argument('--n_best_size', type=int, default=20)
    parser.add_argument('--null_score_diff_threshold', type=float, default=0.0)
    
    # Straggle
    def csv_ints(s: str) -> List[int]:
        if not s: return []
        try: return [int(x) for x in re.split(r"\s*,\s*", s) if x]
        except ValueError: raise argparse.ArgumentTypeError("Expected comma-separated list of integers")
    
    parser.add_argument("--straggle_points", type=int, default=0)
    parser.add_argument("--straggle_prob", type=float, default=0)
    parser.add_argument("--straggle_ranks", type=csv_ints, default=[])
    parser.add_argument("--straggle_amount", type=float, default=0)
    parser.add_argument("--straggle_multiply", type=float, nargs=2, default=[1.0, 1.0])
    parser.add_argument("--straggle_verbose", action='store_true')
    
    args = parser.parse_args()
    
    # Environment setup
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        set_seed(args.seed + args.rank)
    else:
        torch.backends.cudnn.benchmark = True
    
    # Device checks
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        if args.rank == 0:
            print("[Info] Using device=cpu because CUDA is not available", flush=True)
    if args.amp and args.device == 'cpu':
        args.amp = False
        if args.rank == 0:
            print("[Info] Disabling AMP because CUDA is not available", flush=True)
    if args.workers < 1:
        args.workers = 1
    
    sys.stdout.reconfigure(line_buffering=True)
    
    setup_ddp(args)
    
    # Print config
    cfg = {k: v for k, v in vars(args).items() if not k.startswith('_')}
    print(json.dumps(cfg, indent=2))
    
    try:
        train(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()