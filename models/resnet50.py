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
import random
import numpy as np

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
        train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=False)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=args.drop_last)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
                                               num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    return train_loader, val_loader


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
        # if torch.distributed.is_available() and torch.distributed.is_initialized():
        #     t = torch.tensor([self.sum, self.count], dtype=torch.float64, device="cpu")
        #     torch.distributed.all_reduce(t)
        #     self.sum, self.count = t.tolist()
        #     self.avg = self.sum / max(1.0, self.count)
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
                # images = images.to(device, non_blocking=True)
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last) if device.type == 'cuda' else images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if args.amp and device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda'):
                    # with torch.cuda.amp.autocast(device_type='cuda'):
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
    return top1.avg, top5.avg, losses.avg

def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    """Perform 1 full pass over the dataset. Return loss, epoch duration, epoch throughput (imgs/sec)"""
    model.train()
    total_loss = 0.0
    samples_seen = 0

    if device.type == 'cuda':
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()  # on current stream
    else:
        start = time.perf_counter()

    for images, targets in dataloader:
        # images = images.to(device, non_blocking=True)
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last) if device.type == 'cuda' else images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        samples_seen += images.size(0)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            # with torch.cuda.amp.autocast():
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

        total_loss += float(loss.item())

    if device.type == 'cuda':
        end.record() 
        end.synchronize()
        duration = start.elapsed_time(end) / 1000.0  # seconds
    else:
        duration = time.perf_counter() - start

    throughput = samples_seen / max(1e-6, duration)
    return total_loss / max(1, len(dataloader)), duration, throughput

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

    # Model
    # model = models.resnet50(num_classes=args.num_classes).to(device)
    # if device.type == "cuda":
    #     model = DDP(model, device_ids=[args.local_rank])
    # else:
    #     model = DDP(model)
    model = models.resnet50(num_classes=args.num_classes)
    if device.type == "cuda":
        model = model.to(device, memory_format=torch.channels_last)
        model = DDP(model, device_ids=[args.local_rank])
    else:
        model = model.to(device)  # keep default NCHW on CPU
        model = DDP(model)


    print("Model initialized.", flush=True)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,  weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp) if device.type == "cuda" else None

    def now(): return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    best_top1 = 0.0
    best_top5 = 0.0

    if args.rank == 0:
        log = {"time": now(), "config": vars(args), "epochs": {}}
        save_log(args.json, log)


    # print("len(train classes) =", len(train_dataset.classes))
    # print("len(val classes)   =", len(val_dataset.classes))
    # assert train_dataset.classes == val_dataset.classes, "train/val class folder names differ"
    
    for epoch in range(args.epochs):
        print(f"[{now()}][Epoch {epoch:03d}] ...")

        epoch_start = time.time()
        train_loader.sampler.set_epoch(epoch)
        
        train_loss, train_time, train_tp = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        top1, top5, val_loss = validate(model, val_loader, device, args)

        # Print epoch summary with learning rate
        current_lr = scheduler.get_last_lr()[0]
        if args.rank == 0:
            epoch_time = time.time() - epoch_start
            epoch_tp = (len(train_loader.dataset) / max(1, args.world_size)) / max(1e-6, epoch_time)
        
            print(f"[{now()}][Epoch {epoch:03d}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} top1={top1:.2f}% top5={top5:.2f}% "
                  f"lr={current_lr:.6f} time={epoch_time:.2f}s tp= ~{epoch_tp:.1f} img/s", flush=True)
            log["epochs"][str(epoch)] = {
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "top1": float(top1),
                "top5": float(top5),
                "lr": float(current_lr),
                "train_time_sec": float(train_time),
                "epoch_time_sec": float(epoch_time),
                "train_throughput_ips": float(train_tp),
                "epoch_throughput_ips": float(epoch_tp)
            }
            save_log(args.json, log)

        # Step the scheduler after evaluation (end of epoch)
        scheduler.step()
        
        if args.rank == 0 and top1 > best_top1: best_top1 = top1
        if args.rank == 0 and top5 > best_top5: best_top5 = top5


    # if args.rank == 0:
        # with open(args.json, "w") as f:
        #     import json
        #     json.dump(log, f, indent=2)

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
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)

    # Ensure the variables torch.distributed expects are present.
    os.environ.setdefault("RANK",        str(args.rank))
    os.environ.setdefault("WORLD_SIZE",  str(args.world_size))
    os.environ.setdefault("MASTER_ADDR", args.master_addr)
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ.setdefault("GLOO_SOCKET_IFNAME", args.iface)
    os.environ.setdefault("NCCL_SOCKET_IFNAME", args.iface)
    os.environ.setdefault("LOCAL_RANK",  str(args.local_rank))

    # Start the process group
    dist.init_process_group(backend=args.backend, init_method="env://", rank=args.rank, world_size=args.world_size, timeout=datetime.timedelta(seconds=30))

    # if torch.cuda.is_available(): torch.cuda.set_device(args.local_rank)
    if args.rank == 0:
        print(f"[DDP] backend={args.backend} world_size={args.world_size} "
              f"master={args.master_addr}:{args.master_port} iface={args.iface} local_rank={args.local_rank}", flush=True)

def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--iface', type=str, default="ens4f0")
    parser.add_argument('--master_addr', type=str, default="42.0.0.1")
    parser.add_argument("--master_port", type=int, default=29500)
    parser.add_argument("--backend", type=str, default="gloo", help="DDP backend (e.g., gloo, nccl)")
    parser.add_argument("--device", type=str, choices=['cuda', 'cpu'], default='cuda')

    parser.add_argument("--deterministic", action='store_true')
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    parser.add_argument("--drop_last", action='store_true', help="Only for validation")
    
    parser.add_argument("--json", type=str, default="resnet50.json", help="Path to JSON run log")
    args = parser.parse_args()



    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    # Args sanity checks/corrections
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        if args.rank == 0: print("[Info] Using device=cpu because CUDA is not available", flush=True)
    if args.amp and args.device == 'cpu':
        args.amp = False
        if args.rank == 0: print("[Info] Disabling AMP because CUDA is not available", flush=True)
    if args.workers < 1:
        if args.rank == 0: print("[Info] Workers requested < 1; using workers=1", flush=True)
        args.workers = 1 


    sys.stdout.reconfigure(line_buffering=True)

    setup_ddp(args)

    if args.deterministic:
        args.seed = 42 + args.rank
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)


    print(json.dumps(vars(args), indent=2))
    try:
        train(args)
    finally:
        cleanup()

if __name__ == '__main__':
    main()