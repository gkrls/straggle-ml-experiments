import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, date2num
import matplotlib.patheffects as pe

# -----------------------------
# Data
# -----------------------------
gpu_data = [
    ("K40",   "2013-10-01",  4.3),
    ("K80",   "2014-11-17",  8.7),
    ("P100",  "2016-04-05",  21.2),
    ("V100",  "2017-05-10", 125.0),
    ("A100",  "2020-05-14", 312.0),
    ("H100",  "2022-08-01", 990.0),
    ("B200",  "2024-03-18", 2250.0),
]
gpu_df = pd.DataFrame(gpu_data, columns=["name", "date", "tflops"])
gpu_df["date"] = pd.to_datetime(gpu_df["date"]).dt.tz_localize(None)
gpu_df = gpu_df.sort_values("date").reset_index(drop=True)

nic_line = [
    ("10GbE",  "2005-04-01",   10.0),
    ("40GbE",  "2011-08-29",   40.0),
    ("100GbE", "2014-11-12",  100.0),
    ("200GbE", "2019-08-26",  200.0),
    ("400GbE", "2022-03-22",  400.0),
    ("800GbE", "2025-05-18",  800.0),
]
nic_df = pd.DataFrame(nic_line, columns=["name", "date", "gbps"])
nic_df["date"] = pd.to_datetime(nic_df["date"]).dt.tz_localize(None)
nic_df = nic_df.sort_values("date").reset_index(drop=True)

# -----------------------------
# Relative improvements (baselines: K40 and 10G)
# -----------------------------
gpu_baseline = float(gpu_df.iloc[0]["tflops"])  # K40
nic_baseline = float(nic_df.iloc[0]["gbps"])    # 10G
gpu_df["improvement_x"] = gpu_df["tflops"] / gpu_baseline
nic_df["improvement_x"] = nic_df["gbps"] / nic_baseline

# -----------------------------
# Styling helpers
# -----------------------------
NVIDIA_GREEN = "#76B900"
ETH_BLUE = "#1f77b4"
SEGMENT_GRAY = "#555555"
LW = 3.0  # line width points

def fmt_tf_value(tf):
    return f"{tf:.1f} TF" if tf < 100 else f"{tf:.0f} TF"


def rounded_badge(ax, x, y, text, facecolor, fontsize=9):
    t = ax.annotate(
        text, xy=(x, y), xycoords="data",
        xytext=(0, 0), textcoords="offset points",
        fontsize=fontsize, fontweight="bold", color="white",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=facecolor, edgecolor=facecolor, linewidth=0),
    )
    return t

def place_model_next_to_badge(fig, ax, x, y, model_text, color, badge_artist, padding_pts=4, fontsize=9):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox_px = badge_artist.get_window_extent(renderer=renderer)
    width_pts = bbox_px.width * 72.0 / fig.dpi
    dx_pts = width_pts/2 + padding_pts
    t = ax.annotate(
        model_text, xy=(x, y), xycoords="data",
        xytext=(dx_pts, 0), textcoords="offset points",
        fontsize=fontsize, color=color, fontweight="normal",
        ha="left", va="center",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")]
    )
    return t

def place_segment_label_simple(ax, x0, y0, x1, y1, text, color, side="above", fontsize=8.5):
    """
    Place a horizontal label at the midpoint of the segment, offset vertically.
    Simple and clean - no rotation, no complex calculations.
    """
    # Midpoint (geometric mean for y on log scale)
    xm = (x0 + x1) / 2.0
    ym = (y0 * y1) ** 0.5
    
    # Vertical alignment based on side
    if side == "above":
        va = "bottom"
        y_offset = ym * 1.15  # 15% above in log space
    else:
        va = "top"
        y_offset = ym * 0.87  # 13% below in log space
    
    # Simple horizontal label with white stroke for readability
    t = ax.text(
        xm, y_offset, text,
        color=color, fontsize=fontsize, fontweight="normal",
        ha="center", va=va,
        path_effects=[pe.withStroke(linewidth=3, foreground="white")]
    )
    return t

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(13.5, 7.2))

# Lines
ax.semilogy(gpu_df["date"], gpu_df["improvement_x"], linewidth=LW, 
            color=NVIDIA_GREEN, label="GPU improvement (× vs K40)", zorder=2)
ax.semilogy(nic_df["date"], nic_df["improvement_x"], linewidth=LW, 
            color=ETH_BLUE, label="Ethernet improvement (× vs 10G)", zorder=2)

# Badges and model names
for _, r in gpu_df.iterrows():
    badge = rounded_badge(ax, r["date"], r["improvement_x"], 
                         fmt_tf_value(r["tflops"]), NVIDIA_GREEN, fontsize=9)
    place_model_next_to_badge(fig, ax, r["date"], r["improvement_x"], 
                             f"({r['name']})", NVIDIA_GREEN, badge, 
                             padding_pts=4, fontsize=9)

for _, r in nic_df.iterrows():
    rounded_badge(ax, r["date"], r["improvement_x"], 
                 r["name"], ETH_BLUE, fontsize=9)

# Segment labels with simplified approach
# GPU labels above the line
xs_gpu = date2num(gpu_df["date"].values)
for i in range(len(xs_gpu)-1):
    x0, x1 = xs_gpu[i], xs_gpu[i+1]
    y0, y1 = float(gpu_df["improvement_x"].iloc[i]), float(gpu_df["improvement_x"].iloc[i+1])
    mult = y1 / y0
    text = f"{mult:.1f}×"
    place_segment_label_simple(ax, x0, y0, x1, y1, text, SEGMENT_GRAY, 
                              side="above", fontsize=8.5)

# Ethernet labels below the line
xs_nic = date2num(nic_df["date"].values)
for i in range(len(xs_nic)-1):
    x0, x1 = xs_nic[i], xs_nic[i+1]
    y0, y1 = float(nic_df["improvement_x"].iloc[i]), float(nic_df["improvement_x"].iloc[i+1])
    mult = y1 / y0
    text = f"{mult:.1f}×"
    place_segment_label_simple(ax, x0, y0, x1, y1, text, SEGMENT_GRAY, 
                              side="below", fontsize=8.5)

# Axes
ax.set_ylabel("Relative improvement (×, log scale)\nBaselines: GPU=K40 (4.29 TF, 2013-10-01), Ethernet=10G (2005-04-01)")
ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
ax.grid(True, which="both", axis="y", alpha=0.3, zorder=1)

ymin = min(gpu_df["improvement_x"].min(), nic_df["improvement_x"].min())
ymax = max(gpu_df["improvement_x"].max(), nic_df["improvement_x"].max())
ax.set_ylim(ymin * 0.8, ymax * 1.25)

ax.legend(loc="upper left")
ax.set_title("GPU vs Ethernet — Relative Improvement Over Time\nSegment multipliers shown parallel to each line")

fig.tight_layout()
plt.show()

# Save the figure
# out_path = "/mnt/user-data/outputs/gpu_ethernet_improved.png"
# fig.savefig(out_path, dpi=180, bbox_inches='tight')
# print("Saved:", out_path)
# plt.close()