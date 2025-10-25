import argparse
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Subset
import sys
import json
import datetime
import time 
import re
import random
import numpy as np
import math

import dpa


# ------------------------- Dataset ------------------------------

def get_dataloaders(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageFolder(root=os.path.join(args.data, "train"), transform=train_transform)
    val_dataset   = ImageFolder(root=os.path.join(args.data, "val"),   transform=val_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=args.drop_last_train)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=args.drop_last_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
                                               num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=args.prefetch_factor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=args.prefetch_factor)
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
        self.max = 0
    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1.0, self.count)
        self.min = min(self.min, val)
        self.max = max(self.max, val)

    def all_reduce(self):
        if dist.is_available() and dist.is_initialized():
            backend = dist.get_backend()
            # NCCL => must use GPU tensor; otherwise CPU is fine
            device = torch.device(f"cuda:{torch.cuda.current_device()}") if backend == dist.Backend.NCCL else torch.device("cpu")
            t = torch.tensor([self.sum, self.count], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            self.sum, self.count = t.cpu().tolist()
            self.avg = self.sum / max(1.0, self.count)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# ------------------------- Train / Eval -------------------------

@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    top1, top5, losses = AverageMeter(), AverageMeter(), AverageMeter()
    criterion = nn.CrossEntropyLoss().to(device)

    def run_validation(dataloader):
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if args.amp and device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))    

    run_validation(loader)
    top1.all_reduce()
    top5.all_reduce()
    losses.all_reduce()

    if len(loader.sampler) * args.world_size < len(loader.dataset):
        aux_val_dataset = Subset(loader.dataset, range(len(loader.sampler) * args.world_size, len(loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(aux_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        run_validation(aux_val_loader)

    return {'loss': losses.avg, 'top1': top1.avg, 'top5': top5.avg }

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    """Perform 1 full pass over the dataset. Return loss"""
    model.train()
    
    # meters
    losses = AverageMeter()
    top1 = AverageMeter() 
    top5 = AverageMeter()
    step_time = AverageMeter()
    data_time = AverageMeter()

    if device.type == 'cuda':
        epoch_start = torch.cuda.Event(enable_timing=True)
        epoch_end   = torch.cuda.Event(enable_timing=True)
        epoch_start.record()  # on current stream
    else:
        epoch_start = time.perf_counter()

    step_start = time.perf_counter()

    samples_seen = 0.0
    for images, targets in dataloader:
        data_time.update(time.perf_counter() - step_start, n=1)

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        samples_seen += images.size(0)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Calculate accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        # Update meters
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))
        
        step_time.update(time.perf_counter() - step_start, n=1)
        step_start = time.perf_counter()

    if device.type == 'cuda':
        epoch_end.record() 
        epoch_end.synchronize()
        duration = epoch_start.elapsed_time(epoch_end) / 1000.0  # seconds
    else:
        duration = time.perf_counter() - epoch_start

    throughput = samples_seen / max(1e-6, duration)

    local_loss = losses.avg
    losses.all_reduce()
    
    return {
        'loss_global' : losses.avg,
        'loss': local_loss,
        'top1': top1.avg,
        'top5': top5.avg,
        'step_time_min': step_time.min,
        'step_time_max': step_time.max,
        'step_time': step_time.avg,
        'data_time': data_time.avg,
        'comp_time': step_time.avg - data_time.avg,
        'epoch_time': duration,
        'throughput': throughput,
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

    # Model - VGG variants
    model = None
    if args.model == 'vgg11': 
        model = models.vgg11(num_classes=args.num_classes).to(device, memory_format=torch.channels_last if device.type == 'cuda' else torch.contiguous_format)
    elif args.model == 'vgg19': 
        model = models.vgg19(num_classes=args.num_classes).to(device, memory_format=torch.channels_last if device.type == 'cuda' else torch.contiguous_format)
    else: 
        raise ValueError(f"Unsupported model: {args.model}")

    model = DDP(model, device_ids=[args.local_rank] if device.type == "cuda" else None, gradient_as_bucket_view=True,
                find_unused_parameters=False, static_graph=args.static_graph)

    # Wrap the model if DPA backend is requested
    if args.backend.startswith("dpa"):
        model = dpa.DDPWrapper(model, straggle = args.world_size, prescale=args.prescale)

    print(f"Model '{args.model}' initialized.", flush=True)

    # Straggle sim
    straggle = dpa.DDPStraggleSim(points=args.straggle_points, prob=args.straggle_prob, amount=args.straggle_amount, ranks=args.straggle_ranks,
                                  multiplier_range=args.straggle_multiply, verbose=args.straggle_verbose)     

    criterion = nn.CrossEntropyLoss().to(device)
    
    # VGG typically uses SGD with standard momentum (not Nesterov)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, foreach=True)
    
    # VGG learning rate schedule
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp) if device.type == "cuda" else None

    def now(): return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    best_top1 = 0.0
    best_top5 = 0.0

    log = {"time": now(), "config": vars(args), "epochs": {}}
    save_log(args.json, log)
    
    for epoch in range(args.epochs):
        print(f"[{now()}][Epoch {epoch:03d}] ...")
        epoch_start = time.time()

        straggle.reset_stats()

        train_loader.sampler.set_epoch(epoch)
        
        # Train for one epoch and get metrics
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)

        # Validate and get metrics
        val_metrics  = validate(model, val_loader, device, args)

        # Calculate total epoch time and overall throughput
        epoch_time = time.time() - epoch_start

        # Print epoch summary with learning rate
        current_lr = scheduler.get_last_lr()[0]

        print(f"[{now()}][Epoch {epoch:03d}] "
                f"train_loss={train_metrics['loss']:.4f} (global={train_metrics['loss_global']:.4f}) "
                f"val_loss={val_metrics['loss']:.4f} "
                f"top1={val_metrics['top1']:.2f}% top5={val_metrics['top5']:.2f}% "
                f"lr={current_lr:.6f} epoch_time={epoch_time:.2f}s step_time={train_metrics['step_time']:.2f} "
                f"(min={train_metrics['step_time_min']:.2f}s, max={train_metrics['step_time_max']:.2f}) tp=~{train_metrics['throughput']:.1f} img/s", 
                f"straggle_events={straggle_sim.get_stats()['num_straggle_events']}", flush=True)
        
        # Combine all metrics into one dictionary for logging
        epoch_metrics = {
            # Training metrics
            "lr": float(current_lr),
            "train_loss": float(train_metrics['loss']),           # Local loss for rank 0
            "train_loss_global": float(train_metrics['loss_global']), # Global average loss
            "train_top1": float(train_metrics['top1']),
            "train_top5": float(train_metrics['top5']),
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
            "val_top1": float(val_metrics['top1']),
            "val_top5": float(val_metrics['top5']),
            
            # straggle-sim
            "straggle": straggle_sim.get_stats() if straggle_sim.active else {}
        }
        
        log["epochs"][str(epoch)] = epoch_metrics
        save_log(args.json, log)
        
        # Track best validation accuracy
        if val_metrics['top1'] > best_top1: 
            best_top1 = val_metrics['top1']
        if val_metrics['top5'] > best_top5: 
            best_top5 = val_metrics['top5']

        # Step the scheduler after evaluation (end of epoch)
        scheduler.step()

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
    args.local_rank  = (args.rank % torch.cuda.device_count()) if torch.cuda.device_count() else 0
    if args.device == 'cuda' and torch.cuda.is_available(): torch.cuda.set_device(args.local_rank)

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

    if args.backend.startswith("dpa"):
        if not args.dpa_conf: raise RuntimeError(f"--dpa_conf required for backend {args.backend}")
        dpa_device = dpa.DPADeviceOptions.from_config(args.dpa_conf)
        dpa_backend = dpa.DPADpdkBackendOptions.from_config(args.dpa_conf)
        pg_options = dpa.ProcessGroupDPADpdkOptions(dpa_device, dpa_backend)
        # pg_options.hint_pinned_tensor_size = max(200_000_000, args.bucket_cap_mb * (2 ** 20) * 4) # observed max around 150-is MB
        # pg_options.hint_pinned_tensor_pool_size = 20                                              # observed count 13
        pg_options.hint_pinned_tensor_size = max(args.hint_tensor_size, args.bucket_cap_mb * (2 ** 20) * 4 if args.bucket_cap_mb is not None else 0)
        pg_options.hint_pinned_tensor_pool_size = args.hint_tensor_count                                                                            
        dist.init_process_group(backend=args.backend, init_method="env://", rank=args.rank, world_size=args.world_size, timeout = datetime.timedelta(seconds=60), pg_options=pg_options)
    else:
        dist.init_process_group(backend=args.backend, init_method="env://", rank=args.rank, world_size=args.world_size, timeout=datetime.timedelta(seconds=60))


    print(
        f"[DDP] backend={args.backend} world_size={args.world_size} "
        f"master={args.master_addr}:{args.master_port} iface={args.iface} local_rank={args.local_rank}",
        flush=True,
    )

def main():
    parser = argparse.ArgumentParser()

    # DDP/System
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--iface', type=str, default="ens4f0")
    parser.add_argument('--master_addr', type=str, default="42.0.0.1")
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument("--backend", type=str, default="gloo", help="DDP backend (e.g., gloo, nccl)")
    parser.add_argument("--dpa_conf", type=str, default=None, help="Path to dpa config.json")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument("--deterministic", action='store_true')
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--json", type=str, default="resnet.json", help="Path to JSON run log")

    # Training/model
    parser.add_argument('--model', type=str, choices=['vgg11', 'vgg19'], help="VGG model", default="vgg11")
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=90)  
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--learning_rate', type=float, default=0.01, help="Initial learning rate (standard for VGG)")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay (L2 regularization)")
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    parser.add_argument("--drop_last_train", action='store_true', help="Drop last from train dataset")
    parser.add_argument("--drop_last_val", action='store_true', help="Drop last from val dataset")
    parser.add_argument("--static_graph", action='store_true', help="Enable static_graph in DDP")
    parser.add_argument("--prefetch_factor", type=int, default=2)

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
    
    parser.add_argument('--prescale', action="store_true", help="Prescale gradients for allreduce")
    parser.add_argument("--bucket_cap_mb", type=int, default=None, help="DDP bucket capacity")
    parser.add_argument("--hint_tensor_size", type=int, default=200000000, help="Hint for allreduce tensor size (bytes)")
    parser.add_argument("--hint_tensor_count", type=int, default=20, help="Hint for number of tensors allreduced, per step")

    args = parser.parse_args()

    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(args.seed + args.rank)
        np.random.seed(args.seed + args.rank)
        torch.manual_seed(args.seed + args.rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + args.seed)
    else:
        torch.backends.cudnn.benchmark = True

    # Args sanity checks/corrections
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

    print(json.dumps(vars(args), indent=2))
    try:
        train(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    main()