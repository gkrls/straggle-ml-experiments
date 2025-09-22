#!/usr/bin/env python3
import os
import sys
import time
import torch
import torch.distributed as dist
import argparse

import dpa

torch.set_printoptions(
    threshold=10,   # summarize when numel() > 10
    edgeitems=3,    # how many from each end to show
    linewidth=120,  # avoid wrapping
    precision=3,
    sci_mode=False
)


def benchmark(args):
    # Set env vars for init
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
        
      
    elif args.backend == "gloo":
        # Handle Gloo with specific network interface
        if args.gloo_socket_ifname:
            os.environ["GLOO_SOCKET_IFNAME"] = args.gloo_socket_ifname
            print(f"[Rank {args.rank}] Using Gloo with interface: {args.gloo_socket_ifname}")
        else:
            print(f"[Rank {args.rank}] Using Gloo with default interface")
        
        dist.init_process_group(backend="gloo", init_method=init_method, rank=args.rank, world_size=args.world_size)
    
    elif args.backend.startswith("nccl"):
        # Handle NCCL with RDMA or TCP
        actual_backend = "nccl"
        
        # Set socket interface for TCP operations (used by all NCCL modes)
        if args.nccl_socket_ifname: os.environ["NCCL_SOCKET_IFNAME"] = args.nccl_socket_ifname
        
        if args.backend == "nccl_tcp":
            # Force TCP mode by disabling InfiniBand
            os.environ["NCCL_IB_DISABLE"] = "1"
            if args.nccl_socket_ifname: print(f"[Rank {args.rank}] NCCL TCP using interface: {args.nccl_socket_ifname}")
            print(f"[Rank {args.rank}] Using NCCL with TCP (IB disabled)")
            
        elif args.backend == "nccl_rdma":
            # Enable RDMA mode (default for NCCL if IB/RoCE available)
            os.environ["NCCL_IB_DISABLE"] = "0"
            # Set IB HCA device if specified
            if args.nccl_ib_hca: os.environ["NCCL_IB_HCA"] = args.nccl_ib_hca
            if args.nccl_ib_hca: print(f"[Rank {args.rank}] NCCL using HCA: {args.nccl_ib_hca}")
            if args.nccl_socket_ifname: print(f"[Rank {args.rank}] NCCL socket interface: {args.nccl_socket_ifname}")
            print(f"[Rank {args.rank}] Using NCCL with RDMA (IB enabled)")
            
        else:  # Plain "nccl" - use default auto-detection
            print(f"[Rank {args.rank}] Using NCCL with auto-detected transport")
        
        # Additional NCCL tuning options
        if args.nccl_debug: os.environ["NCCL_DEBUG"] = "INFO"
        
        dist.init_process_group(backend=actual_backend, init_method=init_method, rank=args.rank, world_size=args.world_size)
    
    else:
        # Standard PyTorch backends (mpi)
        raise ValueError(f"Unknown backend: {args.backend}")
    
    print(f"[Rank {args.rank}] Initialized {args.backend}... ")
    dist.barrier()
    print(f"[Rank {args.rank}] {args.world_size} ranks ready...")

    # Setup device
    device = torch.device(args.device)
    if args.device == "cuda":
        if not torch.cuda.is_available():  raise RuntimeError("CUDA not available")
        torch.cuda.set_device(args.rank % torch.cuda.device_count())
    
    # Create ALL tensors upfront
    print(f"[Rank {args.rank}] Creating tensors...")
    dtype = torch.float32 if args.type == "float32" else torch.int32
    tensors = [torch.ones(args.size, dtype=dtype, device=device) * (args.rank + 1) * -(i + 1) for i in range(args.warmup + args.iters)]
    # tensors = [torch.full((args.size,), args.rank + 1, dtype=dtype, device=device) for _ in range(args.warmup + args.iters)]

    # Print the inputs
    for i in range(args.warmup + args.iters): print(f"[Rank {args.rank}] Input {i}:", tensors[i], "(warmup)" if i < args.warmup else "")

    # for i in range(args.iters):
    #     t = tensors[args.warmup + i]
    #     print(f"[Rank {args.rank}] Tensor {i}: contiguous={t.is_contiguous()}, last 10 values={t[-10:].tolist()}")
    #     # print(tensors[args.warmup + i])

    dist.barrier()
    
    # DPA context options
    dpa_ctx = {"quantization": int(args.dpa_qnt), "averaging": args.dpa_avg, "prescaled": args.dpa_pre, "pipes": args.dpa_pipes, "straggle": args.world_size}

    def run_allreduce(t):
        if args.backend.startswith("dpa"):
            with dpa.DataplaneContext(**dpa_ctx):
                return dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
        else:
            return dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
    
    # Warmup
    print(f"[Rank {args.rank}] Running {args.warmup} warmup jobs...")
    for i in range(args.warmup): run_allreduce(tensors[i])
  
    # if args.device == "cuda": torch.cuda.synchronize()
    
    # Batch mode - fire all, sync once (like DDP)
    print(f"[Rank {args.rank}] Running {args.iters} timed jobs...")
    if args.batch:
        works = []
        if args.device == "cuda":
            # torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for i in range(args.iters): works.append(run_allreduce(tensors[args.warmup + i]))
            for w in works: w.wait() # Wait for all operations to complete BEFORE recording end time
            # torch.cuda.synchronize()  # Ensure all ops done
            end.record()  # Now record the end event
            torch.cuda.synchronize()  # Sync before measuring
            total_time = start.elapsed_time(end) / 1000.0
            # end.record()
            # torch.cuda.synchronize() # Make sure all the copies etc are finished
            # total_time = start.elapsed_time(end) / 1000.0
        else:
            start = time.perf_counter()
            for i in range(args.iters): works.append(run_allreduce(tensors[args.warmup + i]))
            for w in works: w.wait() # Wait for all operations to complete BEFORE measuring end time
            total_time = time.perf_counter() - start
        
        avg_time = total_time / args.iters
        times = [avg_time]
    
    # Per-iteration mode (sync each)
    else:
        times = []
        for i in range(args.iters):
            dist.barrier()
            
            if args.device == "cuda":
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                run_allreduce(tensors[args.warmup + i]).wait()
                end.record()
                torch.cuda.synchronize()
                elapsed = start.elapsed_time(end) / 1000.0
            else:
                start = time.perf_counter()
                run_allreduce(tensors[args.warmup + i]).wait()
                elapsed = time.perf_counter() - start
            
            times.append(elapsed)
            if args.rank == 0 and args.verbose: print(f"  Iter {i+1}: {elapsed*1000:.2f} ms")
    
    # Print the results
    # for i in range(args.warmup + args.iters): print(f"[Rank {args.rank}] Output {i}:", tensors[i], "(warmup)" if i < args.warmup else "")
    for i in range(args.warmup + args.iters): print(f"[Rank {args.rank}] Output {i}:", tensors[i], "(warmup)" if i < args.warmup else "")

    # Results
    # if args.rank == 0:
    avg_time = sum(times) / len(times)
    bytes_per_elem = 4  # Both float32 and int32 are 4 bytes
    mb = args.size * bytes_per_elem / 1e6
    
    print(f"\n{'='*50}")
    print(f"Backend: {args.backend}")
    if args.backend.startswith("nccl"):
        transport = "RDMA" if args.backend == "nccl_rdma" else "TCP" if args.backend == "nccl_tcp" else "auto"
        print(f"NCCL Transport: {transport}")
    elif args.backend == "gloo" and args.gloo_socket_ifname:
        print(f"Gloo Interface: {args.gloo_socket_ifname}")
    print(f"Device: {args.device}")
    print(f"Data Type: {args.type}")
    print(f"Size: {args.size} elements ({mb:.2f} MB)")
    print(f"Mode: {'batch (single sync)' if args.batch else 'per-iteration'}")
    if args.backend.startswith("dpa"):
        print(f"DPA: quant={args.dpa_qnt}, avg={args.dpa_avg}, pipes={args.dpa_pipes}, prescaled={args.dpa_pre}")
    print(f"{'='*50}")
    print(f"Avg time: {avg_time*1000:.4f} ms")
    print(f"Bandwidth: {mb*8/avg_time:.3f} Mb/s (per rank)")
    print(f"Throughput: {mb/avg_time:.3f} MB/s (per rank)")
    print(f"Elements/s: {args.size/avg_time:.3f} (per rank)")
    
    # Calculate aggregate bandwidth for all-reduce
    # In all-reduce, each rank sends and receives (N-1)/N of the data
    effective_data = mb * 2 #* (args.world_size - 1) / args.world_size
    print(f"Aggregate bandwidth: {effective_data*8/avg_time:.3f} Mb/s")
    print(f"{'='*50}\n")


    dist.all_reduce(tensors[0])
    print(tensors[0])
    
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AllReduce Benchmark")
    
    # Core arguments
    parser.add_argument("-b", "--backend", required=True, choices=["gloo", "nccl", "nccl_rdma", "nccl_tcp", "dpa_sock", "dpa_dpdk"], help="Communication backend to use")
    parser.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("-t", "--type", default="float32", choices=["float32", "int32"], help="Data type for tensors")
    parser.add_argument("-s", "--size", type=int, default=1000000, help="Number of elements in tensor")
    parser.add_argument("-i", "--iters", type=int, default=10, help="Number of iterations")
    parser.add_argument("-w", "--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--batch", action="store_true",  help="Queue all ops, single sync (like DDP)")
    
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

    # ignore user. just enable quant if floats or disable if not
    args.dpa_qnt = args.type == "float32"
    
    # Validation
    if args.backend in ["nccl", "nccl_rdma", "nccl_tcp"] and args.device == "cpu": raise ValueError("NCCL backends require --device cuda")
    
    benchmark(args)