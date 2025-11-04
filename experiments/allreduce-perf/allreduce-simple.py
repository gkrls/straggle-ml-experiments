#!/usr/bin/env python3
import os
import sys
import time
import torch
import torch.distributed as dist
import argparse
import numpy as np

import dpa

torch.set_printoptions(
    threshold=10,   # summarize when numel() > 10
    edgeitems=3,    # how many from each end to show
    linewidth=120,  # avoid wrapping
    precision=3,
    sci_mode=False
)

PATTERN = {
    1: lambda args: torch.ones(args.size, dtype=args.dtype, device=torch.device(args.device)),
    2: lambda args: torch.arange(args.size, device=torch.device(args.device)).remainder(5).add(1).to(args.dtype)
}

def bencmarks(args):
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["RANK"] = str(args.rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)

    # Initialize process group
    init_method = f"tcp://{args.master_addr}:{args.master_port}"
    if args.backend.startswith("dpa"):
        if not dpa or not args.dpa_conf: raise RuntimeError(f"DPA module and --dpa_conf required for {args.backend}")
        
        # Load DPA config
        dpdk = args.backend == "dpa_dpdk"
        device = dpa.DPADeviceOptions.from_config(args.dpa_conf)
        backend = dpa.DPADpdkBackendOptions.from_config(args.dpa_conf) if dpdk else dpa.DPASocketBackendOptions.from_config(args.dpa_conf)
        pg_options = dpa.ProcessGroupDPADpdkOptions(device, backend) if dpdk else dpa.ProcessGroupDPASocketOptions(device, backend) 

        print(f"[Rank {args.rank}] DPA: {backend.addr}:{backend.port}, backend: {args.backend}")

        os.environ["GLOO_SOCKET_IFNAME"] = backend['iface']
        dist.init_process_group(backend=args.backend, init_method=init_method, 
                                rank=args.rank, world_size=args.world_size, pg_options=pg_options)
    
    device = torch.device(args.device)
    if args.device == 'cuda': torch.cuda.set_device(args.rank % torch.cuda.device_count())

    tensor = [PATTERN[args.pattern](args) for i in range(args.warmup + args.iters)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AllReduce Benchmark")
    
    # Core arguments
    parser.add_argument("-b", "--backend", required=True, choices=["gloo", "nccl", "nccl_rdma", "nccl_tcp", "dpa_sock", "dpa_dpdk"], help="Communication backend to use")
    parser.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("-t", "--type", default="float32", choices=["float32", "int32"], help="Data type for tensors")
    parser.add_argument("-s", "--size", type=int, default=1000000, help="Number of elements in tensor")
    parser.add_argument("-p", "--pattern", type=int, default=1, choices=[1,2], help="Select tensor pattern 1=ones, 2=repeating range")
    parser.add_argument("-i", "--iters", type=int, default=10, help="Number of iterations")
    parser.add_argument("-w", "--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--batch", action="store_true",  help="Queue all ops, single sync (like DDP)")

    parser.add_argument("--verify", action="store_true", help="Verify output. Only works for SUM operation (no averaging/prescaling)")
    
    # Statistics aggregation
    parser.add_argument("--global_stats", action="store_true", help="Also compute and report global statistics across all ranks")
    
    # Distributed arguments
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)), help="Rank of this process")
    parser.add_argument("--world_size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)), help="Total number of processes")
    parser.add_argument("--master_addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1"), help="Master node address")
    parser.add_argument("--master_port", type=int, default=int(os.environ.get("MASTER_PORT", 29500)), help="Master node port")
    
    # NCCL specific arguments
    parser.add_argument("--nccl_socket_ifname", help="Network interface for NCCL TCP/socket operations (e.g., ens4f1, eth0)")
    parser.add_argument("--nccl_ib_hca", help="RDMA HCA device for NCCL RDMA mode (e.g., mlx5_0, mlx5_1)")
    parser.add_argument("--nccl_debug", action="store_true", help="Enable NCCL debug output")
    
    # Gloo specific arguments
    parser.add_argument("--gloo_socket_ifname", help="Network interface for Gloo backend (e.g., ens4f1, eth0)")
    
    # DPA specific arguments
    parser.add_argument("--dpa_conf", help="DPA config file")
    parser.add_argument("--dpa_qnt", action="store_true", help="Quantization (single exponent)")
    parser.add_argument("--dpa_avg", action="store_true", help="Averaging")
    parser.add_argument("--dpa_pre", action="store_true", help="Prescaling")
    parser.add_argument("--dpa_pipes", type=int, default=2, help="Number of pipes")
    
    args = parser.parse_args()
    args.dtype = torch.float32 if args.type == "float32" else torch.int32
    args.dpa_qnt = args.type == "float32"     # ignore user. just enable quant if floats or disable if not
    
    # Validation
    if args.backend in ["nccl", "nccl_rdma", "nccl_tcp"] and args.device == "cpu": raise ValueError("NCCL backends require --device cuda")
    if args.device == "cuda" and not torch.cuda.is_available():  raise RuntimeError("CUDA not available")

    benchmark(args)