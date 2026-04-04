#!/usr/bin/env python3
"""nic_monitor.py — sysfs polling, works for RDMA/DPDK/kernel"""
import time
import sys
import os

device = sys.argv[1] if len(sys.argv) > 1 else "mlx5_0"
port = sys.argv[2] if len(sys.argv) > 2 else "1"
output = sys.argv[3] if len(sys.argv) > 3 else "/tmp/nic_throughput.csv"
interval = 0.0002  # 200µs

base = f"/sys/class/infiniband/{device}/ports/{port}/counters"
rx_path = f"{base}/port_rcv_data"
tx_path = f"{base}/port_xmit_data"

# these counters are in 32-bit words (4 bytes each)
def read_counter(path):
    with open(path) as f:
        return int(f.read()) * 4  # convert to bytes

samples = []
t0 = time.monotonic()

try:
    while True:
        rx = read_counter(rx_path)
        tx = read_counter(tx_path)
        samples.append((time.monotonic() - t0, rx, tx))
        time.sleep(interval)
except KeyboardInterrupt:
    pass

with open(output, 'w') as f:
    f.write("time_us,rx_bytes,tx_bytes\n")
    for t, rx, tx in samples:
        f.write(f"{t*1e6:.1f},{rx},{tx}\n")

print(f"{len(samples)} samples written to {output}")