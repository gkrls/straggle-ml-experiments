# -*- coding: utf-8 -*-
"""
GPU vs Ethernet — pace of improvement and compute-to-network mismatch

What this does:
  1) Plots absolute GPU TFLOPs and Ethernet Gb/s over time (log-y), on twin y-axes.
  2) Plots the "FLOPs per byte of network" ratio for each GPU at its release date.
  3) Prints doubling times for both GPU compute and Ethernet line rate.

Notes:
  - GPU metric is "peak training throughput" per era:
      P100 uses FP16 (no Tensor Cores), later parts use FP16/BF16 Tensor Core peak.
  - Ethernet metric is NIC line rate (Gb/s).
  - Dates are public intro/availability-style dates (good enough for the pace story).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# -----------------------------
# GPU data (start at P100; consistent "training throughput" story)
# -----------------------------
gpu_data = [
    ("P100",  "2016-04-05",  21.2,  "FP16 (no TC)"),
    ("V100",  "2017-05-10", 125.0,  "FP16 Tensor"),
    ("A100",  "2020-05-14", 312.0,  "FP16/BF16 Tensor"),
    ("H100",  "2022-08-01", 990.0,  "FP16/BF16 Tensor"),
    ("B200",  "2024-03-18", 2250.0, "FP16/BF16 Tensor (per GPU)"),
]
gpu_df = pd.DataFrame(gpu_data, columns=["name", "date", "tflops", "precision_note"])
gpu_df["date"] = pd.to_datetime(gpu_df["date"])
gpu_df = gpu_df.sort_values("date").reset_index(drop=True)

# -----------------------------
# Ethernet NIC "top-end" line rates (monotonic)
# -----------------------------
nic_line = [
    ("10GbE",  "2005-04-01",   10),
    ("40GbE",  "2011-08-29",   40),
    ("100GbE", "2014-11-12",  100),
    ("200GbE", "2019-08-26",  200),
    ("400GbE", "2022-03-22",  400),
    ("800GbE", "2025-05-18",  800),
]
nic_df = pd.DataFrame(nic_line, columns=["name", "date", "gbps"])
nic_df["date"] = pd.to_datetime(nic_df["date"])
nic_df = nic_df.sort_values("date").reset_index(drop=True)

# -----------------------------
# Helper: doubling time from a log fit
# -----------------------------
def doubling_time(dates: pd.Series, values: pd.Series) -> float:
    """
    Returns doubling time in days using a linear fit of log2(values) vs time.
    """
    dates = pd.to_datetime(dates)
    x_days = (dates - dates.iloc[0]).dt.total_seconds() / (3600 * 24)
    y = np.log2(values / values.iloc[0])
    # Handle small series
    if len(values) < 2 or (values <= 0).any():
        return float("inf")
    m, _ = np.polyfit(x_days, y, 1)  # slope in log2 per day
    return (1 / m) if m > 0 else float("inf")

# -----------------------------
# Pair each GPU with fastest Ethernet available at or before its date
# -----------------------------
paired = pd.merge_asof(
    gpu_df.sort_values("date"),
    nic_df.sort_values("date")[["date", "gbps", "name"]].rename(
        columns={"name": "nic_name", "date": "nic_date"}
    ),
    left_on="date",
    right_on="nic_date",
    direction="backward"
)

# FLOPs per byte of network:
#   (TFLOPs * 1e12 FLOP/s) / (Gb/s * 1e9 bits/s / 8) = TFLOPs * 8000 / Gb/s
paired["flops_per_byte"] = paired["tflops"] * 8000.0 / paired["gbps"]

# -----------------------------
# Figure 1: Absolute values on log-y (twin y-axes)
# -----------------------------
fig, ax1 = plt.subplots(figsize=(11.5, 6.5))

gpu_line = ax1.semilogy(
    gpu_df["date"], gpu_df["tflops"],
    marker="o", linewidth=3.0, markersize=9, label="GPU peak training TFLOPs"
)
for _, r in gpu_df.iterrows():
    ax1.annotate(
        f"{r['name']} • {r['tflops']:.0f} TFLOPs",
        (r["date"], r["tflops"]),
        xytext=(6, 8), textcoords="offset points", fontsize=9
    )
ax1.set_ylabel("GPU TFLOPs (log scale)")
ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m"))

ax2 = ax1.twinx()
nic_line2 = ax2.semilogy(
    nic_df["date"], nic_df["gbps"],
    marker="s", linewidth=3.0, markersize=9, label="Ethernet line rate (Gb/s)"
)
for _, r in nic_df.iterrows():
    ax2.annotate(
        r["name"], (r["date"], r["gbps"]),
        xytext=(6, -14), textcoords="offset points", fontsize=9
    )
ax2.set_ylabel("Ethernet Gb/s (log scale)")

# Build a combined legend
lines = gpu_line + nic_line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left")

ax1.set_title("GPU Compute vs. Ethernet NIC Bandwidth — Pace of Improvement (log scale)")
ax1.grid(True, which="both", axis="y", alpha=0.3)
fig.tight_layout()

# -----------------------------
# Figure 2: Compute-to-network mismatch ratio
# -----------------------------
fig2, ax = plt.subplots(figsize=(11.5, 4.4))
ax.semilogy(
    paired["date"], paired["flops_per_byte"],
    marker="^", linewidth=2.6, markersize=8
)
for _, r in paired.iterrows():
    ax.annotate(
        f"{r['name']} (vs {r['nic_name']})",
        (r["date"], r["flops_per_byte"]),
        xytext=(6, 8), textcoords="offset points", fontsize=9
    )
ax.set_title("Compute-to-Network Mismatch (↑ means compute outpaces network)")
ax.set_ylabel("FLOPs per byte of network (log scale)")
ax.set_xlabel("Date")
ax.grid(True, which="both", axis="y", alpha=0.3)
ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
fig2.tight_layout()

# -----------------------------
# Doubling times + quick summary
# -----------------------------
gpu_dt_days = doubling_time(gpu_df["date"], gpu_df["tflops"])
nic_dt_days = doubling_time(nic_df["date"], nic_df["gbps"])

print(f"GPU doubling time ≈ {gpu_dt_days/30.44:.1f} months")
print(f"NIC doubling time ≈ {nic_dt_days/365.25:.1f} years")
print("\nNotes:")
print("- GPU metric: vendor peak training throughput per era (P100 FP16; V100+ Tensor FP16/BF16).")
print("- Ethernet metric: NIC line rate (Gb/s).")
print("- The mismatch curve uses the fastest Ethernet available at each GPU's release date.")
plt.show()
