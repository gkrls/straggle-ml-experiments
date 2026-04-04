# Pipelining Strategies for Host-Staged In-Network AllReduce

## 1. Background

When performing distributed training across GPU nodes connected by Ethernet, the AllReduce collective requires moving gradient data between GPU memory and the network. Frameworks like NCCL handle this transparently, but custom backends — such as those leveraging in-network computation on programmable switches — must manage the data path explicitly.

This report describes two complementary pipelining strategies for a custom AllReduce backend that uses **host-staged transfers**: data must travel from GPU memory to pinned host memory (D2H), across the network via the switch, and back to the GPU (H2D). We refer to these as **cross-operation pipelining** (optimizing throughput across many AllReduce calls) and **intra-operation pipelining** (optimizing latency of a single AllReduce call).

All measurements in this report were collected on a 6-node cluster connected by a 100 Gbps RoCE network, using 100 MB tensors (25M float32 elements) and NVIDIA GPUs without GPUDirect RDMA.

## 2. The Data Path Problem

For a host-staged AllReduce of a 100 MB tensor, the data path consists of three stages:

```
┌──────────┐       ┌──────────┐       ┌──────────┐
│  D2H     │──────▶│ Network  │──────▶│  H2D     │
│ GPU→Host │       │ Allreduce│       │ Host→GPU │
│  ~8 ms   │       │  ~6 ms   │       │  ~8 ms   │
└──────────┘       └──────────┘       └──────────┘
```

Executed naively (all three stages in sequence), a single AllReduce takes approximately **22 ms**. The question is how to overlap these stages to reduce wall-clock time.

For comparison, NCCL achieves **18.5 ms** per operation under the same conditions. NCCL uses a ring algorithm with fine-grained internal chunking, aggressively overlapping PCIe transfers with network communication — even without GPUDirect RDMA.

## 3. Cross-Operation Pipelining (Throughput Mode)

### 3.1 Core Idea

Rather than overlapping stages within a single operation, this approach overlaps stages **across consecutive operations**. While operation N is in-flight on the network, operation N+1 begins copying its data from GPU to host.

This is effective when the training framework (e.g., DDP) issues multiple AllReduce calls in sequence — either via gradient bucketing or explicit batching.

### 3.2 How It Works

The implementation uses a work-stealing pattern with two key properties:

1. The AllReduce constructor performs a **blocking D2H copy** and returns.
2. A worker thread submits the operation to the backend via `AllReduceAsync`, which **returns immediately** (the backend queues internally). The worker then installs a completion callback for the H2D copy and releases the submission gate.

This means the benchmark (or DDP) calling thread is free to construct the next AllReduce — which triggers its own blocking D2H — while the previous operation's network transfer is still in-flight.

### 3.3 Timeline

For a batch of 4 AllReduce operations with a single synchronization at the end:

```
Main thread:    [█ D2H₀ █][█ D2H₁ █][█ D2H₂ █][█ D2H₃ █]  ...  [wait]
                     │          │          │          │
Workers:          submit₀   submit₁   submit₂   submit₃
                     │          │          │          │
Backend:        [███ net₀ ███][███ net₁ ███][███ net₂ ███][███ net₃ ███]
                              │             │             │
Callbacks:                 [H2D₀]       [H2D₁]       [H2D₂]       [H2D₃]
```

The critical overlap is vertical: while the main thread blocks on D2H₁, the backend is processing net₀ on the NIC. The PCIe bus and the network are busy simultaneously, despite using simple blocking copies.

### 3.4 Why Blocking D2H Is Sufficient

An unintuitive aspect of this design is that the D2H copy is synchronous (`non_blocking=false`). This works because the pipeline is driven by the **calling loop**, not by asynchronous CUDA operations:

- D2H takes ~8 ms for a 100 MB tensor.
- Network transfer takes ~6 ms.
- As long as D2H of the next operation takes longer than (or comparable to) the current network transfer, the NIC is never idle.

Making D2H asynchronous (`non_blocking=true`) would add CUDA event synchronization overhead, thread scheduling latency, and condvar wake-up costs — all for no benefit, since the main thread has nothing else to do while waiting.

### 3.5 Results

With 20 × 100 MB operations issued in a batch:

| Metric | Value |
|---|---|
| Average per-op time | 17.0 ms |
| Throughput | **60 Gbps** (on 100G link) |

The NIC is utilized approximately 60% of the link's theoretical bandwidth, with PCIe copy overhead accounting for most of the remaining gap.

### 3.6 Limitation

This approach does not improve the latency of a **single** AllReduce. When only one operation is in flight (e.g., DDP with a very large bucket), the full serial cost of D2H + network + H2D is exposed:

| Metric | Value |
|---|---|
| Single-op latency | **29.2 ms** |
| NCCL single-op latency | 18.5 ms |

There is no next operation to overlap with, so the pipeline has no effect.


## 4. Intra-Operation Pipelining (Latency Mode)

### 4.1 Core Idea

Instead of treating the tensor as a monolithic block, split it into **N chunks** and pipeline the D2H, network, and H2D stages of each chunk. This overlaps PCIe and network transfers **within a single operation**.

This is conceptually similar to what NCCL does internally with its proxy thread and chunked ring/tree algorithm.

### 4.2 How It Works

The tensor is divided into N equal chunks (e.g., 4 × 25 MB for a 100 MB tensor). Two CUDA streams are used: one for D2H copies, one for H2D copies. The worker thread orchestrates the pipeline:

```cpp
// Kick off D2H for chunk 0
cudaMemcpyAsync(cpu, gpu, chunk_bytes, D2H, d2h_stream);
record(d2h_done[0], d2h_stream);

for (int c = 0; c < num_chunks; c++) {
    // 1. Wait for this chunk's D2H
    cudaEventSynchronize(d2h_done[c]);

    // 2. Start D2H for next chunk (runs on GPU DMA engine)
    if (c + 1 < num_chunks) {
        cudaMemcpyAsync(...chunk c+1..., D2H, d2h_stream);
        record(d2h_done[c+1], d2h_stream);
    }

    // 3. Network allreduce this chunk (blocks CPU thread,
    //    but D2H of c+1 runs in parallel on GPU DMA engine)
    task = AllReduceAsync(chunk_c);
    task->wait();

    // 4. Start H2D for this chunk (runs in parallel with next iteration)
    cudaMemcpyAsync(...chunk c..., H2D, h2d_stream);
}
```

### 4.3 Timeline

For 4 chunks of 25 MB each:

```
d2h_stream: [D2H₀][D2H₁  ][D2H₂  ][D2H₃  ]
                │       │        │        │
CPU thread: [wait₀][wait₁  ][wait₂  ][wait₃  ]
                │       │        │        │
Backend:    [net₀  ][net₁  ][net₂  ][net₃  ]
                │       │        │
h2d_stream:  [H2D₀  ][H2D₁  ][H2D₂  ][H2D₃]
```

Each vertical column shows what happens concurrently:
- While the backend processes chunk N on the network, the GPU DMA engine copies chunk N+1 from GPU to host **and** copies chunk N−1 from host back to GPU.

### 4.4 The Overlap

Without chunking, the total time is the **sum** of all three stages:

```
Total = D2H + Network + H2D ≈ 8 + 6 + 8 = 22 ms
```

With chunking, the stages overlap and the time approaches:

```
Total ≈ D2H₀ + max(D2H, Network, H2D) × N_chunks + H2D_last
```

With enough chunks, this converges toward the time of the slowest stage — plus a small startup and drain cost.

### 4.5 Two CUDA Streams

A subtle but important detail: D2H and H2D copies use **separate CUDA streams**. Most NVIDIA GPUs have independent DMA copy engines for each direction. By placing D2H and H2D on different streams, both can execute concurrently on the PCIe bus:

```
PCIe (Device→Host):  [D2H chunk 2][D2H chunk 3]
PCIe (Host→Device):  [H2D chunk 0][H2D chunk 1]
                      ───── concurrent ─────
```

A single stream would serialize these, eliminating half the overlap.

### 4.6 Results

With 4 chunks on a single 100 MB AllReduce:

| Metric | Value |
|---|---|
| Single-op latency | **18.6 ms** |
| NCCL single-op latency | 18.5 ms |

This brings single-operation performance to parity with NCCL.

### 4.7 Limitation

In batched/throughput scenarios, chunked mode performs worse than cross-operation pipelining:

| Metric | Cross-op | Chunked |
|---|---|---|
| Batched throughput (20 × 100MB) | **60 Gbps** | 47 Gbps |
| Single-op latency (100MB) | 29.2 ms | **18.6 ms** |

The throughput regression occurs because each chunk calls `AllReduceAsync` followed by `task->wait()`, which means the backend processes only one chunk at a time. Between operations, the submission gate prevents any overlap. The NIC has idle gaps.


## 5. Comparison

The two approaches optimize for fundamentally different scenarios:

```
                   Cross-Op Pipelining          Intra-Op Pipelining
                   ════════════════════          ═══════════════════
Overlaps:          D2H[op N+1] with             D2H[chunk N+1] with
                   Network[op N]                Network[chunk N]

Pipeline depth:    Across multiple ops          Within a single op

Best for:          Many AllReduce calls         Single large AllReduce
                   (DDP small buckets)          (DDP large buckets)

Backend queue:     Always full                  Always depth 1
                   (AllReduceAsync returns       (wait() after each chunk)
                    immediately)

NIC utilization:   High (60 Gbps)              Moderate (47 Gbps batched)

Single-op:         Poor (29.2 ms)              Excellent (18.6 ms)
```

### 5.1 Why Each Approach Loses at the Other's Strength

**Why cross-operation pipelining has poor single-op latency:**

This approach only overlaps D2H of the *next* operation with network of the *current* one. When there is no next operation — a single AllReduce followed by a synchronization — there is nothing to overlap with. The full serial cost is exposed:

```
Single op, cross-op mode:

[████ D2H 8ms ████][████ net 6ms ████][████ H2D 8ms ████]
                                                          ▲
                                              Total: ~22 ms (measured: 29 ms
                                              with additional backend overhead)
```

The NIC sits idle during D2H and H2D. No amount of batching cleverness helps because there is only one operation.

**Why intra-operation pipelining has lower batched throughput:**

This approach calls `task->wait()` on every chunk. That means the backend processes exactly one chunk at a time — its internal queue never has more than one item. Between chunks, there are small gaps (event synchronization, function call overhead). Between *operations*, the gap is larger: the submission gate blocks the next operation until the entire current pipeline (all chunks' D2H + network + H2D) completes.

```
Batched, intra-op mode (simplified):

Op 0:  [D2H₀][net₀][D2H₁+H2D₀][net₁][H2D₁]  ←gate held→  Op 1: [D2H₀][net₀]...
                                                ▲
                                      NIC idle during gate
                                      transfer + next D2H
```

Compare with cross-operation mode, where `AllReduceAsync` returns immediately and the backend's internal queue absorbs the next operation without waiting:

```
Batched, cross-op mode:

Backend queue:  [██ net₀ ██][██ net₁ ██][██ net₂ ██][██ net₃ ██]
                           ▲           ▲           ▲
                     No gaps — backend always has work queued
```

The fundamental difference: cross-operation mode keeps the backend's queue full (high throughput, NIC always busy), while intra-operation mode keeps the queue at depth 1 (lower throughput, but PCIe and network overlap within each operation).

### 5.2 When Each Approach Wins

**Cross-operation pipelining** is preferred when the framework issues many AllReduce calls in succession — the common case during DDP training with default or small gradient bucket sizes. The backend's internal queue stays saturated, maximizing NIC utilization.

**Intra-operation pipelining** is preferred when few large AllReduce calls are issued — for example, with very large DDP bucket sizes, or in frameworks that aggregate all gradients into a single operation. It is also preferred during the final synchronization step of any training iteration, where there is no subsequent operation to overlap with.


## 6. Toward a Unified Approach

An ideal implementation would combine both strategies: chunk a single operation to overlap D2H/network/H2D internally, while also allowing the next operation to begin its D2H during the current operation's final chunks.

This requires the submission gate to release earlier — not after the entire chunked pipeline completes, but after the last chunk is submitted to the backend. The next operation's D2H could then overlap with the current operation's final network transfers and H2D copies.

This is architecturally feasible but requires careful coordination between the chunked pipeline, the backend's internal queue, and the submission ordering mechanism. It represents a natural next step.


## 7. The GPUDirect RDMA Opportunity

Both approaches described above are constrained by the host-staged data path: every byte travels over PCIe twice (GPU→Host and Host→GPU). On PCIe Gen3 ×16 (~12 GB/s), a 100 MB round-trip costs approximately 16 ms of DMA time alone.

GPUDirect RDMA eliminates both copies by allowing the NIC to read/write GPU memory directly:

```
Current:    GPU ──PCIe──▶ Host ──NIC──▶ Switch ──NIC──▶ Host ──PCIe──▶ GPU
GPUDirect:  GPU ──────────NIC──▶ Switch ──NIC──────────▶ GPU
```

With GPUDirect, the AllReduce latency would approach the **network time alone** (~6 ms), and both pipelining strategies described in this report would become unnecessary for the data transfer stages. The primary remaining optimization would be overlapping the AllReduce with backward-pass computation — which is handled at the DDP framework level rather than the backend level.