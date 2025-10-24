# Update: segment labels placed slightly OFF the line, oriented parallel to the segment,
# with lighter text weight, and Ethernet recolored to a blue tone.
# Saves to /mnt/data/fig1_gpu_ethernet_log_segment_labels_offset_blue.png

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, date2num
import matplotlib.patheffects as pe

# -----------------------------
# Data
# -----------------------------
gpu_data = [
    ("K40",   "2013-10-01", 4.29,  False),
    ("K80",   "2014-11-17", 8.73,  False), 
    ("P100",  "2016-04-05", 21.2,  False),
    ("V100",  "2017-05-10", 125.0,  True),
    ("A100",  "2020-05-14", 312.0,  True),
    ("H100",  "2022-08-01", 990.0,  True),
    ("B200",  "2024-03-18", 2250.0, True),
]
gpu_df = pd.DataFrame(gpu_data, columns=["name", "date", "tflops", "has_tensorcore"])
gpu_df["date"] = pd.to_datetime(gpu_df["date"]).dt.tz_localize(None)
gpu_df = gpu_df.sort_values("date").reset_index(drop=True)

nic_line = [
    ("10GbE",  "2005-04-01",   10),
    ("40GbE",  "2011-08-29",   40),
    ("100GbE", "2014-11-12",  100),
    ("200GbE", "2019-08-26",  200),
    ("400GbE", "2022-03-22",  400),
    ("800GbE", "2025-05-18",  800),
]
nic_df = pd.DataFrame(nic_line, columns=["name", "date", "gbps"])
nic_df["date"] = pd.to_datetime(nic_df["date"]).dt.tz_localize(None)
nic_df = nic_df.sort_values("date").reset_index(drop=True)

# -----------------------------
# Styling
# -----------------------------
NVIDIA_GREEN = "#76B900"
ETH_BLUE = "#1f77b4"  # blue-ish

def label_segments_logy(ax, dates, ys, color, fmt="{:.2f}x", fontsize=9, offset_pts=10):
    """
    Place a label like '2.5x' near the midpoint of each segment, rotated parallel to the line
    and offset *off* the line by `offset_pts` in the perpendicular direction.
    """
    xs = date2num(dates.dt.to_pydatetime())
    for i in range(len(xs)-1):
        x0, x1 = xs[i], xs[i+1]
        y0, y1 = float(ys.iloc[i]), float(ys.iloc[i+1])
        if y0 <= 0 or y1 <= 0:
            continue
        # Text
        mult = y1 / y0
        text = fmt.format(mult)
        # Midpoint (geometric mean for y since log scale)
        xm = (x0 + x1) / 2.0
        ym = (y0 * y1) ** 0.5
        # Rotation: slope in log10 space
        dy = np.log10(y1) - np.log10(y0)
        dx = x1 - x0
        angle = np.degrees(np.arctan2(dy, dx)) if dx != 0 else 0.0
        # Offset in points along the normal (angle + 90°)
        theta = np.deg2rad(angle + 90.0)
        dx_pts = offset_pts * np.cos(theta)
        dy_pts = offset_pts * np.sin(theta)
        ax.annotate(
            text,
            xy=(xm, ym), xycoords="data",
            xytext=(dx_pts, dy_pts), textcoords="offset points",
            color=color, fontsize=fontsize, fontweight="normal",
            ha="center", va="center",
            rotation=angle, rotation_mode="anchor",
            path_effects=[pe.withStroke(linewidth=3, foreground="white")]
        )

def boxed_label(ax, x, y, text, facecolor, offset=(6, 8), fontsize=9, ha="left", va="center"):
    bbox = dict(boxstyle="square,pad=0.25", facecolor=facecolor, edgecolor=facecolor, linewidth=0)
    try:
        ax.annotate(
            text, xy=(x, y), xytext=offset, textcoords="offset points",
            fontsize=fontsize, fontweight="bold", color="white", ha=ha, va=va, bbox=bbox
        )
    except Exception:
        bbox["boxstyle"] = "round,pad=0.25"
        ax.annotate(
            text, xy=(x, y), xytext=offset, textcoords="offset points",
            fontsize=fontsize, fontweight="bold", color="white", ha=ha, va=va, bbox=bbox
        )

# -----------------------------
# Figure: Two log-y axes with off-line segment labels (parallel to segments)
# -----------------------------
fig, ax1 = plt.subplots(figsize=(12.5, 6.6))

# GPU
ax1.semilogy(
    gpu_df["date"], gpu_df["tflops"],
    marker="o", linewidth=3.2, markersize=9, color=NVIDIA_GREEN, label="GPU peak training TFLOPs"
)
for i, r in gpu_df.iterrows():
    # star = "*" if r["has_tensorcore"] else ""
    label = f"{r['name']}\n{r['tflops']:.0f}TF"
    offset = (8, 10 if i % 2 == 0 else -18)
    boxed_label(ax1, r["date"], r["tflops"], label, facecolor=NVIDIA_GREEN, offset=offset, fontsize=9)
label_segments_logy(ax1, gpu_df["date"], gpu_df["tflops"], color=NVIDIA_GREEN, fmt="{:.2f}x", fontsize=9, offset_pts=12)

ax1.set_ylabel("GPU TFLOPs (log scale)")
ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
ax1.grid(True, which="both", axis="y", alpha=0.3)

# Ethernet (blue-ish)
ax2 = ax1.twinx()
ax2.semilogy(
    nic_df["date"], nic_df["gbps"],
    marker="s", linewidth=3.2, markersize=9, color=ETH_BLUE, label="Ethernet line rate (Gb/s)"
)
for j, r in nic_df.iterrows():
    g_label = f"{int(r['gbps'])}G"
    offset = (8, -14 if j % 2 == 0 else 12)
    boxed_label(ax2, r["date"], r["gbps"], g_label, facecolor=ETH_BLUE, offset=offset, fontsize=9)
label_segments_logy(ax2, nic_df["date"], nic_df["gbps"], color=ETH_BLUE, fmt="{:.2f}x", fontsize=9, offset_pts=12)

ax2.set_ylabel("Ethernet Gb/s (log scale)")

# Legend & title
lines = ax1.get_lines() + ax2.get_lines()
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left")
ax1.set_title("GPU vs Ethernet (log-y) — Off-line Segment Labels (parallel to segment)")

fig.text(0.01, 0.01, "* GPUs with Tensor Cores", fontsize=9)
fig.tight_layout()
out_path = "/mnt/data/fig1_gpu_ethernet_log_segment_labels_offset_blue.png"
# fig.savefig(out_path, dpi=180)

plt.show()

# print("Saved:", out_path)
