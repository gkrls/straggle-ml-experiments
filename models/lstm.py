import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from datasets import load_dataset
from collections import Counter
import string
import sys
import json
import datetime
import time
import random
import numpy as np

# ------------------------- Dataset Classes ------------------------------

def simple_tokenizer(text):
    """Basic whitespace and punctuation tokenizer"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

def build_vocab(dataset_name, max_vocab=20000):
    """Build vocabulary from training data"""
    if dataset_name == 'sst2':
        dataset = load_dataset("glue", "sst2", split="train")
        text_field = "sentence"
    elif dataset_name == 'imdb':
        dataset = load_dataset("imdb", split="train")
        text_field = "text"
    elif dataset_name == 'ag_news':
        dataset = load_dataset("ag_news", split="train")
        text_field = "text"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    counter = Counter()
    for example in dataset:
        tokens = simple_tokenizer(example[text_field])
        counter.update(tokens)
    
    vocab = {"<pad>": 0, "<unk>": 1}
    for i, (word, _) in enumerate(counter.most_common(max_vocab - 2), start=2):
        vocab[word] = i
    return vocab

class TextClassificationDataset(Dataset):
    """Generic text classification dataset wrapper"""
    def __init__(self, dataset_name, split, vocab, max_len=None):
        self.vocab = vocab
        self.dataset_name = dataset_name
        
        # Load dataset
        if dataset_name == 'sst2':
            self.data = load_dataset("glue", "sst2", split=split)
            self.text_field = "sentence"
            self.label_field = "label"
            self.max_len = max_len or 64
        elif dataset_name == 'imdb':
            self.data = load_dataset("imdb", split=split)
            self.text_field = "text"
            self.label_field = "label"
            self.max_len = max_len or 256
        elif dataset_name == 'ag_news':
            # AG News uses different split names
            if split == "validation":
                split = "test"  # AG News doesn't have validation, use test
            self.data = load_dataset("ag_news", split=split)
            self.text_field = "text"
            self.label_field = "label"
            self.max_len = max_len or 128
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def __len__(self):
        return len(self.data)
    
    def encode(self, text):
        tokens = simple_tokenizer(text)
        if len(tokens) == 0:
            tokens = ["<unk>"]
        ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens][:self.max_len]
        length = len(ids)
        ids = ids + [0] * (self.max_len - length)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(length, dtype=torch.long)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        x, length = self.encode(item[self.text_field])
        return x, torch.tensor(item[self.label_field]), length

def get_dataloaders(args, vocab):
    """Create train and validation dataloaders"""
    train_dataset = TextClassificationDataset(args.dataset, "train", vocab, max_len=args.max_len)
    
    # For AG News, use test split as validation
    val_split = "test" if args.dataset == "ag_news" else "validation"
    val_dataset = TextClassificationDataset(args.dataset, val_split, vocab, max_len=args.max_len)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=args.drop_last_train)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=args.drop_last_val)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=args.prefetch_factor)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=args.prefetch_factor)
    
    return train_loader, val_loader

def get_dataset_config(dataset_name):
    """Get dataset-specific configuration"""
    configs = {
        'sst2': {
            'num_classes': 2,
            'max_len': 64,
            'expected_acc': '81-85%',
            'epochs': 15,
            'batch_size': 32
        },
        'imdb': {
            'num_classes': 2,
            'max_len': 256,
            'expected_acc': '85-88%',
            'epochs': 10,
            'batch_size': 32
        },
        'ag_news': {
            'num_classes': 4,
            'max_len': 128,
            'expected_acc': '90-92%',
            'epochs': 10,
            'batch_size': 64
        }
    }
    return configs.get(dataset_name, configs['sst2'])

# ------------------------- Model --------------------------------

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, dropout_prob=0.5, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                           batch_first=True, bidirectional=bidirectional,
                           dropout=dropout_prob if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Adjust output size based on bidirectional
        lstm_output_size = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_size, num_classes)
    
    def forward(self, x, lengths):
        embeds = self.embedding(x)
        embeds = self.dropout(embeds)
        
        packed = nn.utils.rnn.pack_padded_sequence(
            embeds, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        
        if self.bidirectional:
            # Concatenate the last hidden states from both directions
            forward_hidden = hidden[-2]
            backward_hidden = hidden[-1]
            out = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            out = hidden[-1]
        
        out = self.dropout(out)
        return self.fc(out)

# ------------------------- Metrics ------------------------------

class AverageMeter:
    def __init__(self):
        self.reset()
    
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

def accuracy(output, target):
    """Computes the accuracy"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = output.argmax(dim=1)
        correct = pred.eq(target).float().sum().item()
        return correct * 100.0 / batch_size

# ------------------------- Train / Eval -------------------------

@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    acc = AverageMeter()
    losses = AverageMeter()
    criterion = nn.CrossEntropyLoss().to(device)
    
    def run_validation(dataloader):
        with torch.no_grad():
            for inputs, targets, lengths in dataloader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                if args.amp and device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = model(inputs, lengths)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs, lengths)
                    loss = criterion(outputs, targets)
                
                batch_acc = accuracy(outputs, targets)
                losses.update(loss.item(), inputs.size(0))
                acc.update(batch_acc, inputs.size(0))
    
    run_validation(loader)
    acc.all_reduce()
    losses.all_reduce()
    
    # Handle remaining samples if dataset doesn't divide evenly
    if len(loader.sampler) * args.world_size < len(loader.dataset):
        aux_val_dataset = Subset(loader.dataset, range(len(loader.sampler) * args.world_size, len(loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        run_validation(aux_val_loader)
    
    return {'val_loss': losses.avg, 'val_acc': acc.avg, 'val_perplexity': np.exp(losses.avg)}

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    """Perform 1 full pass over the dataset"""
    model.train()
    
    losses = AverageMeter()
    acc = AverageMeter()
    step_time = AverageMeter()
    data_time = AverageMeter()
    
    if device.type == 'cuda':
        epoch_start = torch.cuda.Event(enable_timing=True)
        epoch_end = torch.cuda.Event(enable_timing=True)
        epoch_start.record()
    else:
        epoch_start = time.perf_counter()
    
    step_start = time.perf_counter()
    samples_seen = 0
    
    for inputs, targets, lengths in dataloader:
        data_time.update(time.perf_counter() - step_start, n=1)
        
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        batch_size = inputs.size(0)
        samples_seen += batch_size
        
        optimizer.zero_grad(set_to_none=True)
        
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs, lengths)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        
        batch_acc = accuracy(outputs, targets)
        losses.update(loss.item(), batch_size)
        acc.update(batch_acc, batch_size)
        
        step_time.update(time.perf_counter() - step_start, n=1)
        step_start = time.perf_counter()
    
    if device.type == 'cuda':
        epoch_end.record()
        epoch_end.synchronize()
        duration = epoch_start.elapsed_time(epoch_end) / 1000.0
    else:
        duration = time.perf_counter() - epoch_start
    
    throughput = samples_seen / max(1e-6, duration)
    
    local_loss = losses.avg
    local_acc = acc.avg
    
    losses.all_reduce()
    acc.all_reduce()
    
    return {
        'train_loss_global': losses.avg,
        'train_loss': local_loss,
        'train_acc_global': acc.avg,
        'train_acc': local_acc,
        'train_perplexity': np.exp(local_loss),
        'train_step_time': step_time.avg,
        'train_data_time': data_time.avg,
        'train_comp_time': step_time.avg - data_time.avg,
        'epoch_duration': duration,
        'epoch_throughput': throughput,
    }

def save_log(path, log):
    """Atomically write log dict to JSON file"""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def train(args):
    device = torch.device(args.device)
    
    # Get dataset-specific configuration
    dataset_config = get_dataset_config(args.dataset)
    
    # Override num_classes based on dataset
    args.num_classes = dataset_config['num_classes']
    
    # Use dataset defaults if not specified
    if args.max_len is None:
        args.max_len = dataset_config['max_len']
    if args.epochs is None:
        args.epochs = dataset_config['epochs']
    if args.batch_size is None:
        args.batch_size = dataset_config['batch_size']
    
    # Build vocabulary and get data loaders
    if args.rank == 0:
        print(f"Building vocabulary for {args.dataset}...", flush=True)
    vocab = build_vocab(args.dataset, max_vocab=args.max_vocab)
    train_loader, val_loader = get_dataloaders(args, vocab)
    
    # Model
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
        dropout_prob=args.dropout,
        bidirectional=args.bidirectional
    ).to(device)
    
    model = DDP(model, device_ids=[args.local_rank] if device.type == "cuda" else None,
                gradient_as_bucket_view=True, find_unused_parameters=False, static_graph=args.static_graph)
    
    if args.rank == 0:
        print(f"\nDataset: {args.dataset.upper()}")
        print(f"  Training samples: {len(train_loader.dataset):,}")
        print(f"  Validation samples: {len(val_loader.dataset):,}")
        print(f"  Vocabulary size: {len(vocab):,}")
        print(f"  Number of classes: {args.num_classes}")
        print(f"  Max sequence length: {args.max_len}")
        print(f"  Expected accuracy: {dataset_config['expected_acc']}")
        print(f"\nModel: LSTM ({'bidirectional' if args.bidirectional else 'unidirectional'}, "
              f"{args.num_layers} layers)")
        print(f"  Embedding dim: {args.embed_dim}")
        print(f"  Hidden dim: {args.hidden_dim}")
        print(f"  Dropout: {args.dropout}")
        print(f"\nTraining:")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Optimizer: Adam")
        print(f"  Scheduler: StepLR (step={args.step_size}, gamma={args.gamma})")
        print("=" * 60, flush=True)
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, foreach=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp) if device.type == "cuda" else None
    
    def now():
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    best_acc = 0.0
    best_perplexity = float('inf')
    
    if args.rank == 0:
        log = {
            "time": now(),
            "dataset": args.dataset,
            "config": vars(args),
            "vocab_size": len(vocab),
            "epochs": {}
        }
        save_log(args.json, log)
    
    for epoch in range(args.epochs):
        if args.rank == 0:
            print(f"[{now()}][Epoch {epoch:03d}] ...", flush=True)
        
        epoch_start = time.time()
        train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_metrics = validate(model, val_loader, device, args)
        
        # Calculate total epoch time
        epoch_time = time.time() - epoch_start
        epoch_throughput = len(train_loader.dataset) / max(1e-6, epoch_time)
        
        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]
        
        if args.rank == 0:
            print(f"[{now()}][Epoch {epoch:03d}] "
                  f"train_loss={train_metrics['train_loss']:.4f} (global={train_metrics['train_loss_global']:.4f}) "
                  f"val_loss={val_metrics['val_loss']:.4f} "
                  f"train_ppl={train_metrics['train_perplexity']:.2f} val_ppl={val_metrics['val_perplexity']:.2f} "
                  f"train_acc={train_metrics['train_acc']:.1f}% val_acc={val_metrics['val_acc']:.1f}% "
                  f"lr={current_lr:.6f} time={epoch_time:.2f}s tp=~{epoch_throughput:.1f} samples/s", flush=True)
            
            # Log metrics
            epoch_metrics = {
                "train_loss": float(train_metrics['train_loss']),
                "train_loss_global": float(train_metrics['train_loss_global']),
                "train_acc": float(train_metrics['train_acc']),
                "train_acc_global": float(train_metrics['train_acc_global']),
                "train_perplexity": float(train_metrics['train_perplexity']),
                "train_step_time": float(train_metrics['train_step_time']),
                "train_data_time": float(train_metrics['train_data_time']),
                "train_comp_time": float(train_metrics['train_comp_time']),
                "train_duration": float(train_metrics['epoch_duration']),
                "train_throughput": float(train_metrics['epoch_throughput']),
                "val_loss": float(val_metrics['val_loss']),
                "val_acc": float(val_metrics['val_acc']),
                "val_perplexity": float(val_metrics['val_perplexity']),
                "lr": float(current_lr),
                "epoch_time": float(epoch_time),
                "epoch_throughput": float(epoch_throughput),
                "steps": int(len(train_loader))
            }
            
            log["epochs"][str(epoch)] = epoch_metrics
            save_log(args.json, log)
            
            # Track best validation accuracy
            if val_metrics['val_acc'] > best_acc:
                best_acc = val_metrics['val_acc']
                # print(f"  New best accuracy: {best_acc:.2f}%", flush=True)
            if val_metrics['val_perplexity'] < best_perplexity:
                best_perplexity = val_metrics['val_perplexity']
        
        # Step the scheduler
        scheduler.step()
    
    if args.rank == 0:
        print(f"\n[{now()}] Training completed!")
        print(f"Best validation accuracy: {best_acc:.2f}%")
        print(f"Best validation perplexity: {best_perplexity:.2f}")
        print(f"Expected accuracy for {args.dataset}: {dataset_config['expected_acc']}", flush=True)

# ------------------------- Entry / Setup ------------------------

def setup_ddp(args):
    def env_int(key, default):
        return default if os.environ.get(key) in (None, "") else int(os.environ.get(key))
    def env_str(key, default):
        return default if os.environ.get(key) in (None, "") else os.environ.get(key)
    
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
    
    # GLOO settings
    os.environ.setdefault("GLOO_SOCKET_IFNAME", args.iface)
    os.environ.setdefault("GLOO_SOCKET_NTHREADS", "8")
    os.environ.setdefault("GLOO_NSOCKS_PERTHREAD", "2")
    os.environ.setdefault("GLOO_BUFFSIZE", "8388608")
    
    # NCCL settings
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

def main():
    parser = argparse.ArgumentParser(description='Distributed LSTM training for text classification')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, choices=['sst2', 'imdb', 'ag_news'], default='sst2',
                       help='Dataset to use for training')
    
    # Distributed training arguments
    parser.add_argument('--rank', type=int, default=0, help='Rank of current process')
    parser.add_argument('--world_size', type=int, default=1, help='Number of processes')
    parser.add_argument('--iface', type=str, default="ens4f0", help='Network interface')
    parser.add_argument('--master_addr', type=str, default="42.0.0.1", help='Master node address')
    parser.add_argument("--master_port", type=int, default=29500, help='Master node port')
    parser.add_argument("--backend", type=str, default="gloo", choices=['gloo', 'nccl'],
                       help="DDP backend")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], default='cuda',
                       help='Device to use for training')
    
    # System arguments
    parser.add_argument("--deterministic", action='store_true', help='Use deterministic algorithms')
    parser.add_argument("--workers", type=int, default=4, help='Number of data loading workers')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=256, help="Embedding dimension")
    parser.add_argument('--hidden_dim', type=int, default=512, help="LSTM hidden dimension")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of LSTM layers")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout probability")
    parser.add_argument('--max_len', type=int, default=None, help="Maximum sequence length (dataset-specific default if not set)")
    parser.add_argument('--max_vocab', type=int, default=20000, help="Maximum vocabulary size")
    parser.add_argument('--bidirectional', action='store_true', help="Use bidirectional LSTM")
    parser.add_argument('--num_classes', type=int, default=None, help="Number of output classes (auto-detected)")
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (dataset-specific default if not set)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size per GPU (dataset-specific default if not set)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 penalty)')
    parser.add_argument('--step_size', type=int, default=5, help="StepLR step size")
    parser.add_argument('--gamma', type=float, default=0.5, help="StepLR gamma")
    
    # Optimization arguments
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    parser.add_argument("--drop_last_train", action='store_true', help="Drop last batch from train dataset")
    parser.add_argument("--drop_last_val", action='store_true', help="Drop last batch from val dataset")
    parser.add_argument("--static_graph", action='store_true', help="Enable static_graph in DDP")
    parser.add_argument("--prefetch_factor", type=int, default=2, help='Prefetch factor for data loader')
    
    # Logging
    parser.add_argument("--json", type=str, default=None, help="Path to JSON run log")
    
    args = parser.parse_args()
    
    # Set default JSON log path based on dataset
    if args.json is None:
        args.json = f"lstm_{args.dataset}.json"
    
    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
    
    # Args sanity checks
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
    
    if args.deterministic:
        args.seed = 42 + args.rank
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    if args.rank == 0:
        print(json.dumps(vars(args), indent=2), flush=True)
    
    try:
        train(args)
    finally:
        cleanup()

if __name__ == '__main__':
    main()