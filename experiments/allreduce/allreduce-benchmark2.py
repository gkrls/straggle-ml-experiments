#!/usr/bin/env python3
from contextlib import nullcontext
from datetime import datetime
import json
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
    2: lambda args: torch.arange(args.size, device=torch.device(args.device)).remainder(5).add(1).to(args.dtype),
    3: lambda args: torch.ones(args.size, dtype=args.dtype, device=torch.device(args.device)) * (args.rank + 1)
}

def init(args):
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.master_port)
    os.environ["RANK"] = str(args.rank)
    os.environ["WORLD_SIZE"] = str(args.world_size)

    init_method = f"tcp://{args.master_addr}:{args.master_port}"
    
    if args.backend.startswith("dpa"):
        if not dpa or not args.dpa_conf: raise RuntimeError(f"DPA module and --dpa_conf required for {args.backend}")
        
        # Load DPA config
        dpdk = args.backend == "dpa_dpdk"
        device = dpa.DPADeviceOptions.from_config(args.dpa_conf)
        backend = dpa.DPADpdkBackendOptions.from_config(args.dpa_conf) if dpdk else dpa.DPASocketBackendOptions.from_config(args.dpa_conf)
        pg_options = dpa.ProcessGroupDPADpdkOptions(device, backend) if dpdk else dpa.ProcessGroupDPASocketOptions(device, backend)
        pg_options.hint_pinned_tensor_size = args.size * 4
        pg_options.hint_pinned_tensor_pool_size = args.warmup + args.iters

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
        raise ValueError(f"Unknown backend: {args.backend}")

    print(f"[Rank {args.rank}] Initialized {args.backend}... ")

def results(args, data):
    
    def make_serializable(obj):
        if isinstance(obj, torch.dtype): return str(obj).split('.')[-1]  # e.g., "float32"
        elif isinstance(obj, torch.device): return str(obj)
        else: return obj

    out = {k: make_serializable(v) for k, v in vars(args).items()}
    out['data'] = data
    # up
    
    # {
    #     'time' : ,
    #     'args' : ,
    #     'data' : data
    # }

    with open(args.json, 'w') as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))

    # Results output
    # print(f"\n{'='*50}")
    # print(f"Backend: {args.backend}")
    # if args.backend.startswith("nccl"):
    #     transport = "RDMA" if args.backend == "nccl_rdma" else "TCP" if args.backend == "nccl_tcp" else "auto"
    #     print(f"NCCL Transport: {transport}")
    # elif args.backend == "gloo" and args.gloo_socket_ifname:
    #     print(f"Gloo Interface: {args.gloo_socket_ifname}")
    # print(f"Device: {args.device}")
    # print(f"Data Type: {args.type}")
    # print(f"Size: {args.size} elements ({data["bytes"]/1e6:.2f} MB)")
    # print(f"Mode: {'batch (single sync)' if args.batch else 'per-iteration'}")
    # if args.backend.startswith("dpa"): print(f"DPA: quant={args.dpa_qnt}, avg={args.dpa_avg}, pipes={args.dpa_pipes}, prescaled={args.dpa_pre}")
    # print(f"{'='*50}")
    
    # # Local results (always printed)
    # print(f"[Rank {args.rank}] Local Results:")
    # if args.batch:
    #     print(f"  Time (ms):      {data['time_mean']:.4f} (batch mode - single aggregate measurement)")
    # else:
    #     print(f"  Time (ms):      mean={data['time_mean']:.4f}, std={data['time_std']:.4f}")
    #     print(f"                  min={data['time_min']:.4f}, max={data['time_max']:.4f}")
    #     print(f"                  p50={data['time_p50']:.4f}, p95={data['time_p95']:.4f}, p99={data['time_p99']:.4f}")
    # print(f"  Throughput:     {data['elem_per_sec']:.0f} elements/sec")
    # print(f"  Bandwidth:      {data['bits_per_sec']/1e9:.3f} GB/s ({data['bits_per_sec']/1e9*8:.3f} Gbps)")
    # if args.global_stats:
    #     # Allreduce all metrics (average everything)
    #     metrics_tensor = torch.tensor([
    #         time_mean, time_std, time_min, time_max, time_p50, time_p95, time_p99,
    #         throughput, bandwidth
    #     ], dtype=torch.float64, device=device)  # Use same device as the allreduce operations
        
    #     dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
    #     metrics_tensor /= args.world_size
        
    #     # Also get global min/max of each worker's mean performance
    #     worker_perf = torch.tensor([time_mean, throughput, bandwidth], dtype=torch.float64, device=device)
    #     worker_min = worker_perf.clone()
    #     worker_max = worker_perf.clone()
    #     dist.all_reduce(worker_min, op=dist.ReduceOp.MIN)
    #     dist.all_reduce(worker_max, op=dist.ReduceOp.MAX)
        
    #     print(f"\nGlobal Results (averaged across {args.world_size} ranks):")
    #     if args.batch:
    #         print(f"  Time (ms):      {metrics_tensor[0]:.4f} (worker min={worker_min[0]:.4f}, max={worker_max[0]:.4f})")
    #     else:
    #         print(f"  Time (ms):      mean={metrics_tensor[0]:.4f}, std={metrics_tensor[1]:.4f}")
    #         print(f"                  min={metrics_tensor[2]:.4f}, max={metrics_tensor[3]:.4f}")
    #         print(f"                  p50={metrics_tensor[4]:.4f}, p95={metrics_tensor[5]:.4f}, p99={metrics_tensor[6]:.4f}")
    #         print(f"                  worker min={worker_min[0]:.4f}, max={worker_max[0]:.4f}")
    #     print(f"  Throughput:     {metrics_tensor[7]:.0f} elements/sec (worker min={worker_min[1]:.0f}, max={worker_max[1]:.0f})")
    #     print(f"  Bandwidth:      {metrics_tensor[8]/1e9:.3f} GB/s ({metrics_tensor[8]/1e9*8:.3f} Gbps)")
    #     print(f"                  worker min={worker_min[2]/1e9:.3f}, max={worker_max[2]/1e9:.3f} GB/s")
    # print(f"{'='*50}\n")

    # def run_allreduce(t):
    #     if args.backend.startswith("dpa"):
    #         with dpa.DataplaneContext(**dpa_ctx):
    #             return dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
    #     else:
    #         return dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=True)
    

def benchmark(args):    
    dist.barrier()
    print(f"[Rank {args.rank}] {args.world_size} ranks ready...")

    # Setup device
    device = torch.device(args.device)
    if args.device == "cuda": torch.cuda.set_device(args.rank % torch.cuda.device_count())
    
    # Create ALL tensors upfront
    print(f"[Rank {args.rank}] Creating tensors...")
    tensors = [PATTERN[args.pattern](args) for i in range(args.warmup + args.iters)]

    # Print the inputs
    for i in range(args.warmup + args.iters): print(f"[Rank {args.rank}] Input {i}:", tensors[i], "(warmup)" if i < args.warmup else "")

    dist.barrier()
    
    # DPA context options
    dpa_ctx = {"quantization": args.dpa_qnt, "averaging": args.dpa_avg, "prescaled": args.dpa_pre, "pipes": args.dpa_pipes, 
               "straggle": args.world_size if not args.straggle_k else args.straggle_k}

    # Batch mode - fire all, sync once (like DDP)
    print(f"[Rank {args.rank}] Running {args.warmup} warmup jobs and {args.iters} timed jobs...")

    op = dist.ReduceOp.AVG if (args.backend.startswith("dpa") and (args.dpa_avg or args.dpa_pre)) else dist.ReduceOp.SUM

    with dpa.DataplaneContext(**dpa_ctx) if args.backend.startswith("dpa") else nullcontext():
        if args.batch:
            jobs = []
            
            # Warmup
            for i in range(args.warmup): jobs.append(dist.all_reduce(tensors[i], op=op, async_op=True))
            for j in jobs: j.wait()
            jobs.clear()
            
            # Timed iterations - use same timing method for both CPU and CUDA
            t_start = time.time_ns()
            for i in range(args.iters):
                if args.straggle_rank == args.rank and args.straggle_num > 0:
                    args.straggle_num -= 1
                    time.sleep(args.straggle_ms / 1000)
                # if args.straggle_ms and args.straggle_rank == args.rank:
                #     if args.straggle_num is None:
                #         time.sleep(args.straggle_ms / 1000)
                #     elif args.straggle_num > 0:
                #         args.straggle_num -= 1
                #         time.sleep(args.straggle_ms / 1000)
                jobs.append(dist.all_reduce(tensors[args.warmup + i], op=op, async_op=True))

            for j in jobs: j.wait()
            # for j in jobs: j.synchronize()
            torch.cuda.synchronize()
            total_time = (time.time_ns() - t_start) / 1e9  # Convert ns to seconds
            
            avg_time = total_time / args.iters
            # For batch mode, we only have one aggregate measurement
            times = [avg_time]
            for i in range(args.warmup + args.iters): 
                print(f"[Rank {args.rank}] Output {i}:", tensors[i], "(warmup)" if i < args.warmup else "")
        # Per-iteration mode (sync each)
        else:
            for i in range(args.warmup):
                dist.all_reduce(tensors[i], op=op, async_op=True).wait()
                # run_allreduce(tensors[i]).wait()

            # Use consistent timing for both CPU and CUDA
            torch.cuda.synchronize()

            times = []
            for i in range(args.iters):
                t_start = time.time_ns()

                if args.straggle_rank == args.rank and args.straggle_num > 0:
                    args.straggle_num -= 1
                    time.sleep(args.straggle_ms / 1000)

                # if args.straggle_ms and args.straggle_rank == args.rank:
                #     if args.straggle_num is None: time.sleep(args.straggle_ms / 1000)
                #     elif args.straggle_num > 0:
                #         args.straggle_num -= 1
                #         time.sleep(args.straggle_ms / 1000)

                dist.all_reduce(tensors[args.warmup + i], op=op, async_op=True).wait()                
                torch.cuda.synchronize()
                
                elapsed = (time.time_ns() - t_start) / 1e9  # Convert ns to seconds
                times.append(elapsed)
                
            for i in range(args.warmup):
                print(f"[Rank {args.rank}] Output {i}:", tensors[i], "(warmup)")
            for i in range(args.iters): 
                print(f"[Rank {args.rank}] Output {i}:", tensors[i], f"({times[i]} ms)")

            
                # if args.rank == 0 and args.verbose: 
                #     print(f"  Iter {i+1}: {elapsed*1000:.2f} ms")

    # Calculate metrics
    bytes_per_elem = 4  # Both float32 and int32 are 4 bytes
    tensor_bytes = args.size * bytes_per_elem
    
    # Network bytes received depends on backend algorithm
    if args.backend.startswith("dpa"):
        # DPA: each node sends the full tensor and receives the full tensor once
        network_bytes_received = tensor_bytes * 2
    else:
        # NCCL/Gloo use ring allreduce: each node receives (N-1)/N of the data
        network_bytes_received = tensor_bytes * (args.world_size - 1) / args.world_size
    
    times_np = np.array(times)
    
    # Compute all local metrics
    time_mean = np.mean(times_np) * 1000  # ms
    
    if args.batch:
        # In batch mode, we only have one measurement
        time_std = 0.0
        time_min = time_mean
        time_max = time_mean
        time_p50 = time_mean
        time_p95 = time_mean
        time_p99 = time_mean
    else:
        # Per-iteration mode has multiple measurements
        time_std = np.std(times_np) * 1000
        time_min = np.min(times_np) * 1000
        time_max = np.max(times_np) * 1000
        time_p50 = np.percentile(times_np, 50) * 1000
        time_p95 = np.percentile(times_np, 95) * 1000
        time_p99 = np.percentile(times_np, 99) * 1000
    
    throughput = args.size / (time_mean / 1000)  # elements/sec
    bandwidth = network_bytes_received / (time_mean / 1000)  # bytes/sec
    
    
    data = {
        "bytes" : tensor_bytes,
        "times" : times,
        "time_mean" : time_mean,
        "time_std"  : time_std,
        "time_min"  : time_min,
        "time_max"  : time_max,
        "time_p50"  : time_p50,
        "time_p95"  : time_p95,
        "time_p99"  : time_p99,
        "elem_per_sec" : args.size / (time_mean / 1000),
        "bits_per_sec" : tensor_bytes * 8 / (time_mean / 1000) / 1e9
    }


    if args.verify:
        if op != dist.ReduceOp.SUM: raise RuntimeError("Verification only supports simple SUM. Disable DPA averaging/prescaling")

        local_ok, first_failure = True, True
        original = PATTERN[args.pattern](args)

        for i, out in enumerate(tensors):
            expected = original * args.world_size
            tol = 1e-5 if args.dtype == torch.float32 else 0
            diff = (out - expected).abs()
            max_err = diff.max().item()
            ok = (max_err <= tol) if out.is_floating_point() else (max_err == 0)
            
            if not ok and local_ok:
                bad = (diff > tol).nonzero(as_tuple=False).flatten()
                idx = bad[:min(10, len(bad))].tolist()  # Show up to 10 errors
                flat_out = out.flatten()
                flat_exp = expected.flatten()
                flat_orig = original.flatten()
                samples = [(j, float(flat_orig[j].item()), float(flat_exp[j].item()), float(flat_out[j].item())) for j in idx]
                first_failure = (f"[Rank {args.rank}] Verification FAILED at tensor {i}: "
                                f"max_err={max_err:.3e}\n"
                                f"  Bad samples (index, original, expected, actual): {samples}")
                local_ok = False

        ok_tensor = torch.tensor(1 if local_ok else 0, device=device, dtype=torch.int32)
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
        if ok_tensor.item() == 1:
            if args.rank == 0: 
                print("✅ Verification PASSED (simple SUM).")
        else:
            if first_failure: 
                print(first_failure)
            if args.rank == 0: 
                print("❌ Verification FAILED (simple SUM). See rank logs above.")

    results(args, data)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AllReduce Benchmark")
    
    # Core arguments
    parser.add_argument("-b", "--backend", required=True, choices=["gloo", "nccl", "nccl_rdma", "nccl_tcp", "dpa_sock", "dpa_dpdk"], help="Communication backend to use")
    parser.add_argument("-d", "--device", default="cpu", choices=["cpu", "cuda"], help="Device to run on")
    parser.add_argument("-t", "--type", default="float32", choices=["float32", "int32"], help="Data type for tensors")
    parser.add_argument("-s", "--size", type=int, default=1000000, help="Number of elements in tensor")
    parser.add_argument("-p", "--pattern", type=int, default=1, choices=[1,2,3], help="Select tensor pattern 1=ones, 2=repeating range, 3=rank")
    parser.add_argument("-i", "--iters", type=int, default=10, help="Number of iterations")
    parser.add_argument("-w", "--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--batch", action="store_true",  help="Queue all ops, single sync (like DDP)")

    parser.add_argument("--verify", action="store_true", help="Verify output. Only works for SUM operation (no averaging/prescaling)")
    parser.add_argument("--json", type=str, default=None)
    
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

    parser.add_argument("--straggle_k", type=int, default=0, help="Straggle K value")
    parser.add_argument("--straggle_ms", type=float, default=0, help="Straggle before each allreduce call")
    parser.add_argument("--straggle_num", type=int, default=2 ** 32 - 1, help="Number of straggles")
    parser.add_argument("--straggle_rank", type=int, default=None, help="Rank to straggle")
    
    args = parser.parse_args()
    args.date = datetime.now().strftime("%B %d, %Y at %I:%M:%S %p")
    args.json = args.json if args.json is not None else os.path.join(os.path.dirname(__file__), "allreduce-benchmark2.json")
    args.dtype = torch.float32 if args.type == "float32" else torch.int32
    args.dpa_qnt = args.type == "float32"     # ignore user. just enable quant if floats or disable if not

    # Validation
    if args.device == "cuda" and not torch.cuda.is_available():  raise RuntimeError("CUDA not available")
    if args.backend in ["nccl", "nccl_rdma", "nccl_tcp"] and args.device == "cpu": raise ValueError("NCCL backends require --device cuda")
    
    init(args)
    benchmark(args)

    dist.destroy_process_group()
