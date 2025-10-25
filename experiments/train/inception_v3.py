import argparse
import os
import re
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
import math

import dpa

# ------------------------- Dataset (Inception-v3 only) ------------------------------

def get_dataloaders(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    img_size = 299  # Inception-v3 standard input
    val_resize = int(img_size * 256 / 224)  # 342

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.2, inplace=True),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(val_resize),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageFolder(root=os.path.join(args.data, "train"), transform=train_transform)
    val_dataset   = ImageFolder(root=os.path.join(args.data, "val"),   transform=val_transform)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True, drop_last=args.drop_last_train)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False, drop_last=args.drop_last_val)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch_factor,
    )
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
        self.max = 0.0
    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(1.0, self.count)
        self.min = min(self.min, float(val))
        self.max = max(self.max, float(val))
    def all_reduce(self):
        if dist.is_available() and dist.is_initialized():
            backend = dist.get_backend()
            device = torch.device(f"cuda:{torch.cuda.current_device()}") if backend == dist.Backend.NCCL else torch.device("cpu")
            t = torch.tensor([self.sum, self.count], dtype=torch.float64, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            self.sum, self.count = t.cpu().tolist()
            self.avg = self.sum / max(1.0, self.count)

def accuracy(output, target, topk=(1,)):
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
                        # In eval mode, inception_v3 returns main logits only
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))

    run_validation(loader)
    top1.all_reduce(); top5.all_reduce(); losses.all_reduce()

    # Cover leftover samples if sampler < dataset
    if len(loader.sampler) * args.world_size < len(loader.dataset):
        aux_val_dataset = Subset(loader.dataset, range(len(loader.sampler) * args.world_size, len(loader.dataset)))
        aux_val_loader = torch.utils.data.DataLoader(aux_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        run_validation(aux_val_loader)

    return {'loss': losses.avg, 'top1': top1.avg, 'top5': top5.avg }


def _compute_inception_loss(outputs, targets, criterion, use_aux: bool, aux_weight: float):
    # Torchvision Inception returns InceptionOutputs(logits, aux_logits) during training when aux is enabled.
    # Standard recipe: total_loss = main_loss + aux_weight * aux_loss (aux_weight ≈ 0.3)
    if use_aux and (hasattr(outputs, 'aux_logits') or (isinstance(outputs, tuple) and len(outputs) == 2)):
        main_out = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        aux_out  = outputs.aux_logits if hasattr(outputs, 'aux_logits') else outputs[1]
        return criterion(main_out, targets) + aux_weight * criterion(aux_out, targets), main_out
    else:
        main_out = outputs.logits if hasattr(outputs, 'logits') else outputs
        return criterion(main_out, targets), main_out


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, args):
    model.train()

    losses = AverageMeter(); top1 = AverageMeter(); top5 = AverageMeter()
    step_time = AverageMeter(); data_time = AverageMeter()

    if device.type == 'cuda':
        epoch_start = torch.cuda.Event(enable_timing=True)
        epoch_end   = torch.cuda.Event(enable_timing=True)
        epoch_start.record()
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
                loss, main_out = _compute_inception_loss(outputs, targets, criterion, args.aux_logits, args.aux_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss, main_out = _compute_inception_loss(outputs, targets, criterion, args.aux_logits, args.aux_weight)
            loss.backward()
            optimizer.step()

        acc1, acc5 = accuracy(main_out, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        step_time.update(time.perf_counter() - step_start, n=1)
        step_start = time.perf_counter()

    if device.type == 'cuda':
        epoch_end.record(); epoch_end.synchronize()
        duration = epoch_start.elapsed_time(epoch_end) / 1000.0
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
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(log, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def build_inception(args, device):
    m = models.inception_v3(
        num_classes=args.num_classes,
        aux_logits=args.aux_logits,
    ).to(device, memory_format=torch.channels_last if device.type == 'cuda' else torch.contiguous_format)
    return m


def train(args):
    device = torch.device(args.device)

    # Data
    train_loader, val_loader = get_dataloaders(args)

    # Model (Inception-v3 only)
    model = build_inception(args, device)

    model = DDP(
        model,
        device_ids=[args.local_rank] if device.type == "cuda" else None,
        gradient_as_bucket_view=True,
        find_unused_parameters=False,
        static_graph=args.static_graph,
    )

    if args.backend.startswith("dpa"):
        model = dpa.DDPWrapper(model, straggle = args.world_size, prescale=args.prescale)

    print(f"Model 'inception_v3' initialized. aux_logits={args.aux_logits}", flush=True)

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    # Optimizer / Scheduler
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True, foreach=True)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, momentum=0.0, alpha=0.9, eps=0.001, weight_decay=1e-5, foreach=True)
        warmup_epochs = 5
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/25, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - warmup_epochs, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        for pg in optimizer.param_groups:
            pg['lr'] = args.learning_rate / 25.0

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp) if device.type == "cuda" else None

    def now(): return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    best_top1 = 0.0; best_top5 = 0.0

    log = {"time": now(), "config": vars(args), "epochs": {}}
    save_log(args.json, log)

    for epoch in range(args.epochs):
        print(f"[{now()}][Epoch {epoch:03d}] ...")

        epoch_start = time.time()
        train_loader.sampler.set_epoch(epoch)

        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, args)

        # Validate
        val_metrics  = validate(model, val_loader, device, args)

        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"[{now()}][Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} (global={train_metrics['loss_global']:.4f}) "
            f"val_loss={val_metrics['loss']:.4f} "
            f"top1={val_metrics['top1']:.2f}% top5={val_metrics['top5']:.2f}% "
            f"lr={current_lr:.6f} epoch_time={epoch_time:.2f}s step_time={train_metrics['step_time']:.2f} "
            f"(min={train_metrics['step_time_min']:.2f}s, max={train_metrics['step_time_max']:.2f}) tp=~{train_metrics['throughput']:.1f} img/s",
            flush=True
        )

        epoch_metrics = {
            "lr": float(current_lr),
            "train_loss": float(train_metrics['loss']),
            "train_loss_global": float(train_metrics['loss_global']),
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
            "val_loss": float(val_metrics['loss']),
            "val_top1": float(val_metrics['top1']),
            "val_top5": float(val_metrics['top5']),
        }

        log["epochs"][str(epoch)] = epoch_metrics
        save_log(args.json, log)

        best_top1 = max(best_top1, val_metrics['top1'])
        best_top5 = max(best_top5, val_metrics['top5'])

        scheduler.step()

# ------------------------- Entry / Setup ------------------------

def setup_ddp(args):
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

    os.environ.setdefault("RANK",        str(args.rank))
    os.environ.setdefault("WORLD_SIZE",  str(args.world_size))
    os.environ.setdefault("MASTER_ADDR", args.master_addr)
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ.setdefault("LOCAL_RANK",  str(args.local_rank))

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

    if args.backend.startswith("dpa"):
        if not args.dpa_conf:
            raise RuntimeError(f"--dpa_conf required for backend {args.backend}")
        dpa_device = dpa.DPADeviceOptions.from_config(args.dpa_conf)
        dpa_backend = dpa.DPADpdkBackendOptions.from_config(args.dpa_conf)
        pg_options = dpa.ProcessGroupDPADpdkOptions(dpa_device, dpa_backend)
        pg_options.hint_pinned_tensor_size = max(200_000_000, args.bucket_cap_mb * (2 ** 20) * 4 if args.bucket_cap_mb is not None else 0)
        pg_options.hint_pinned_tensor_pool_size = 20
        dist.init_process_group(backend=args.backend, init_method="env://", rank=args.rank, world_size=args.world_size, timeout = datetime.timedelta(seconds=60), pg_options=pg_options)
    else:
        dist.init_process_group(backend=args.backend, init_method="env://", rank=args.rank, world_size=args.world_size, timeout=datetime.timedelta(seconds=60))

    print(f"[DDP] backend={args.backend} world_size={args.world_size} "
          f"master={args.master_addr}:{args.master_port} iface={args.iface} local_rank={args.local_rank}", flush=True)


def main():
    parser = argparse.ArgumentParser()

    # DDP/System
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--iface', type=str, default="ens4f0")
    parser.add_argument('--master_addr', type=str, default="42.0.0.1")
    parser.add_argument('--master_port', type=int, default=29500)
    parser.add_argument('--backend', type=str, default='gloo', help='DDP backend (e.g., gloo, nccl)')
    parser.add_argument('--dpa_conf', type=str, default=None, help='Path to dpa config.json')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--json', type=str, default='inception_v3.json', help='Path to JSON run log')

    # Inception-only training
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.18, help='Initial LR (RMSProp path warms up)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--amp', action='store_true', help='Enable mixed precision on CUDA')
    parser.add_argument('--drop_last_train', action='store_true', help='Drop last from train dataset')
    parser.add_argument('--drop_last_val', action='store_true', help='Drop last from val dataset')
    parser.add_argument('--static_graph', action='store_true', help='Enable static_graph in DDP')
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--prescale', action='store_true', help='Prescale gradients for allreduce')
    parser.add_argument('--bucket_cap_mb', type=int, default=None, help='DDP bucket capacity')

    parser.add_argument('--optimizer', type=str, choices=['sgd', 'rmsprop'], default='rmsprop')

    # Inception-specific options
    parser.add_argument('--aux_logits', action='store_true', help='Enable auxiliary logits branch during training')
    parser.add_argument('--aux_weight', type=float, default=0.3, help='Aux logits loss weight')

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

    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
        if args.rank == 0:
            print('[Info] Using device=cpu because CUDA is not available', flush=True)
    if args.amp and args.device == 'cpu':
        args.amp = False
        if args.rank == 0:
            print('[Info] Disabling AMP because CUDA is not available', flush=True)
    if args.workers < 1:
        if args.rank == 0:
            print('[Info] Workers requested < 1; using workers=1', flush=True)
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
