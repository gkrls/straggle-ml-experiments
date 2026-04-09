import json
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from itertools import cycle
from pathlib import Path
from matplotlib.patches import Patch

with open(Path(__file__).parent / "bench.json") as f:
    data = json.load(f)

tp_gpu = data["allreduce-gpu"]["throughput"]
tp_sz  = data["allreduce-tensor-size"]["throughput"]
lat_gpu = data["allreduce-gpu"]["latency"]
lat_sz  = data["allreduce-tensor-size"]["latency"]

def avg_gbit(runs):
    if not runs: return None, 0.0
    g = [r["gbit_per_sec"] for r in runs]
    return np.mean(g), (np.std(g) if len(g) > 1 else 0.0)

def avg_latency(run):
    if not run or "time_mean" not in run:
        return None, 0.0
    return run["time_mean"], run.get("time_std", 0.0)

def gbit_std_from_run(run):
    """Extract (gbit, std) from a single tensor-size run dict.
    Computes per-iteration gbit/s from the times array if available."""
    if not isinstance(run, dict) or "bytes" not in run:
        return None, 0.0
    byt = run["bytes"]
    times = run.get("times", [])
    if len(times) > 1:
        gbits = [(byt * 8 / t_ms / 1e6) for t_ms in times]
        return np.mean(gbits), np.std(gbits)
    return run.get("gbit_per_sec", 0.0), 0.0

def latency_std_from_run(run):
    """Extract (latency_ms, std) from a single tensor-size run dict."""
    if not isinstance(run, dict) or "bytes" not in run:
        return None, 0.0
    return run.get("time_mean", 0.0), run.get("time_std", 0.0)

# ── config ──
THROUGHPUT_BARS = [
    "dpa-256/8-t", "dpa-256/6-t", "dpa-64/8-t", "dpa-64/6-t",
    "nccl-rdma-ring-gdr/8", "nccl-rdma-tree-gdr/8", "nccl-tcp-ring",
    "nccl-tcp-tree", "gloo",
]

LATENCY_BARS = [
    "dpa-256/8-l", "dpa-256/6-l", "dpa-64/8-l", "dpa-64/6-l",
    "nccl-rdma-ring-gdr/8", "nccl-rdma-tree-gdr/8", "nccl-tcp-ring",
    "nccl-tcp-tree", "gloo",
]

THROUGHPUT_LINE = [
    "nccl-rdma-ring",
    "dpa-256/6",
    # "dpa-256/6-i",
]

LATENCY_LINE = [
    "nccl-rdma-ring",
    "dpa-256/6-l",
    "dpa-256/6-t",
]

OVERLAYS = {
    "nccl-rdma-tree-gdr/1": [("nccl-rdma-tree-gdr/4", "--")],
    "nccl-rdma-ring-gdr/1": [("nccl-rdma-ring-gdr/4", "--")],
}

MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*", "h"]
LINESTYLES = ["-", "--", "-.", ":"]

def color_of(k):
    if k == "gloo":              return "#8C7E78"
    if k.startswith("nccl-tcp"): return "#C4843E" if "tree" in k else "#D4A261"
    if k.startswith("nccl-rdma"):
        return "#2E5645" if "ring" in k else "#5B8373"
    if k.startswith("dpa"):
        return {"dpa-64": "#8BADC7", "dpa-128": "#5A8DB5", "dpa-256": "#2D6A9F"}.get(
            k.split("/")[0], "#5A8DB5")
    return "#888"

def latex_color(k):
    if k == "gloo":              return "cG"
    if k.startswith("nccl-tcp"): return "cT"
    if k.startswith("nccl-rdma"):
        return "cRr" if "ring" in k else "cRt"
    if k.startswith("dpa"):
        return {"dpa-64": "cD1", "dpa-128": "cD1", "dpa-256": "cD2"}.get(
            k.split("/")[0], "cD1")
    return "black"

def latex_mark(k):
    if "rdma" in k and "ring" in k: return "square*", f"fill={latex_color(k)}"
    return "*", f"fill={latex_color(k)}"

def line_styles(keys):
    mk = cycle(MARKERS)
    ls = cycle(LINESTYLES)
    out = {}
    for k in keys:
        out[k] = dict(color=color_of(k), marker=next(mk), ms=4, ls=next(ls))
    return out

# ── LaTeX printers ──
def print_latex_bars(title, keys, vals, errs):
    print(f"\n%% ── {title} ──")
    for i, (k, v, e) in enumerate(zip(keys, vals, errs)):
        c = latex_color(k)
        print(f"  \\addplot[fill={c},draw=black!40,line width=0.1pt,")
        print(f"    error bars/.cd,y dir=both,y explicit]")
        print(f"    coordinates{{({i},{v:.2f}) +- (0,{e:.2f})}}; % {k}")

def print_latex_line(title, key, sizes_mb, values, errs):
    c = latex_color(key)
    mk, mk_opts = latex_mark(key)
    dashed = ",dashed" if key.endswith("-i") else ""
    coords = "".join(f"({s:.2f},{v:.2f}) +- (0,{e:.2f})"
                     for s, v, e in zip(sizes_mb, values, errs))
    print(f"\n%% ── {title}: {key} ──")
    print(f"  \\addplot[{c},mark={mk},mark options={{{mk_opts},scale=0.6}}{dashed},")
    print(f"    error bars/.cd,y dir=both,y explicit]")
    print(f"    coordinates{{{coords}}};")

# ── figure ──
plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0), dpi=150)
W = 0.72

# ── bar chart helpers ──
def plot_throughput(ax, size, title):
    keys, vals, errs = [], [], []
    for k in THROUGHPUT_BARS:
        runs = tp_gpu.get(k, {}).get("data", {}).get(size, [])
        if not runs: continue
        m, s = avg_gbit(runs)
        keys.append(k); vals.append(m); errs.append(s)
    x = np.arange(len(keys))
    bars = ax.bar(x, vals, W, yerr=errs, capsize=2,
                  color=[color_of(k) for k in keys], edgecolor='#222', linewidth=0.5,
                  error_kw=dict(lw=0.7, capthick=0.5), zorder=3)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+errs[i]+0.6,
                f'{v:.1f}', ha='center', va='bottom', fontsize=5, fontweight='bold', color='#333')
    for i, k in enumerate(keys):
        for ovl_key, ls in OVERLAYS.get(k, []):
            runs = tp_gpu.get(ovl_key, {}).get("data", {}).get(size, [])
            if not runs: continue
            m, _ = avg_gbit(runs)
            ax.hlines(m, x[i]-W/2, x[i]+W/2, ls=ls, colors='#8B1A1A', lw=2, zorder=5)
    ax.set_title(title, fontweight='bold', fontsize=9, pad=6)
    ax.set_ylabel("Throughput (Gbit/s)")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=40, ha='right', fontsize=6)
    ax.set_xlim(-0.6, len(keys)-0.3)
    ax.set_axisbelow(True); ax.yaxis.grid(True, color='#DDD', lw=0.5)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    print_latex_bars(title, keys, vals, errs)
    return max(vals)*1.15 if vals else 1

def plot_latency(ax, size, title):
    keys, vals, errs = [], [], []
    for k in LATENCY_BARS:
        run = lat_gpu.get(k, {}).get("data", {}).get(size)
        if not run: continue
        m, s = avg_latency(run)
        if m is None: continue
        keys.append(k); vals.append(m); errs.append(s)
    if not keys:
        ax.set_title(title, fontweight='bold', fontsize=9, pad=6)
        ax.text(0.5, 0.5, "(no data)", ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='#999')
        return 1
    x = np.arange(len(keys))
    bars = ax.bar(x, vals, W, yerr=errs, capsize=2,
                  color=[color_of(k) for k in keys], edgecolor='#222', linewidth=0.5,
                  error_kw=dict(lw=0.7, capthick=0.5), zorder=3)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+errs[i]+0.3,
                f'{v:.2f}', ha='center', va='bottom', fontsize=5, fontweight='bold', color='#333')
    for i, k in enumerate(keys):
        for ovl_key, ls in OVERLAYS.get(k, []):
            run = lat_gpu.get(ovl_key, {}).get("data", {}).get(size)
            if not run: continue
            m, _ = avg_latency(run)
            if m is None: continue
            ax.hlines(m, x[i]-W/2, x[i]+W/2, ls=ls, colors='#8B1A1A', lw=2, zorder=5)
    ax.set_title(title, fontweight='bold', fontsize=9, pad=6)
    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(x); ax.set_xticklabels(keys, rotation=40, ha='right', fontsize=6)
    ax.set_xlim(-0.6, len(keys)-0.3)
    ax.set_axisbelow(True); ax.yaxis.grid(True, color='#DDD', lw=0.5)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    print_latex_bars(title, keys, vals, errs)
    return max(vals)*1.15 if vals else 1

# ── top row: throughput bars ──
ymax_tp = 0
for size, ax, t in [("100", axes[0,0], "100 MB"),
                     ("25", axes[0,1], "25 MB")]:
    ymax_tp = max(ymax_tp, plot_throughput(ax, size, t))
for a in axes[0,:2]: a.set_ylim(0, ymax_tp)

# ── top-right: throughput vs tensor size ──
ax = axes[0, 2]
tp_line_sty = line_styles(THROUGHPUT_LINE)
for k in THROUGHPUT_LINE:
    sty = tp_line_sty[k]
    sizes_mb, gbits, errs = [], [], []
    for sz_str, run in sorted(tp_sz.get(k, {}).items(), key=lambda x: float(x[0])):
        g, s = gbit_std_from_run(run)
        if g is None: continue
        sizes_mb.append(run["bytes"] / 1e6); gbits.append(g); errs.append(s)
    if sizes_mb:
        ax.errorbar(sizes_mb, gbits, yerr=errs, label=k, lw=1.5,
                    capsize=2, capthick=0.5, elinewidth=0.7,
                    color=sty["color"], marker=sty["marker"],
                    ms=sty["ms"], ls=sty["ls"])
        print_latex_line("Throughput vs tensor size", k, sizes_mb, gbits, errs)
ax.set_xscale("log"); ax.set_xlabel("Tensor size (MB)")
ax.set_ylabel("Throughput (Gbit/s)")
ax.set_title("Throughput vs tensor size", fontweight='bold', fontsize=9, pad=6)
ax.legend(fontsize=6, loc='lower right')
ax.set_axisbelow(True); ax.yaxis.grid(True, color='#DDD', lw=0.5)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

# ── bottom row: latency bars ──
ymax_lat = 0
for size, ax, t in [("100", axes[1,0], "Latency: 100 M elem (100 MB)"),
                     ("25", axes[1,1], "Latency: 25 M elem (25 MB)")]:
    ymax_lat = max(ymax_lat, plot_latency(ax, size, t))
for a in axes[1,:2]: a.set_ylim(0, ymax_lat)

# ── bottom-right: latency vs tensor size ──
ax = axes[1, 2]
has_data = False
lat_line_sty = line_styles(LATENCY_LINE)
for k in LATENCY_LINE:
    sty = lat_line_sty[k]
    sizes_mb, lats, errs = [], [], []
    for sz_str, run in sorted(lat_sz.get(k, {}).items(), key=lambda x: float(x[0])):
        m, s = latency_std_from_run(run)
        if m is None: continue
        sizes_mb.append(run["bytes"] / 1e6); lats.append(m); errs.append(s)
    if sizes_mb:
        ax.errorbar(sizes_mb, lats, yerr=errs, label=k, lw=1.5,
                    capsize=2, capthick=0.5, elinewidth=0.7,
                    color=sty["color"], marker=sty["marker"],
                    ms=sty["ms"], ls=sty["ls"])
        has_data = True
        print_latex_line("Latency vs tensor size", k, sizes_mb, lats, errs)
ax.set_xscale("log"); ax.set_xlabel("Tensor size (MB)")
ax.set_ylabel("Latency (ms)")
ax.set_title("Latency vs tensor size", fontweight='bold', fontsize=9, pad=6)
if has_data:
    ax.legend(fontsize=6, loc='upper left')
else:
    ax.text(0.5, 0.5, "(no data)", ha='center', va='center',
            transform=ax.transAxes, fontsize=10, color='#999')
ax.set_axisbelow(True); ax.yaxis.grid(True, color='#DDD', lw=0.5)
for sp in ['top','right']: ax.spines[sp].set_visible(False)

# ── legend ──
legend_h = [
    Patch(fc='#8C7E78', ec='#222', lw=0.5, label='Gloo'),
    Patch(fc='#C4843E', ec='#222', lw=0.5, label='NCCL/TCP'),
    Patch(fc='#5B8373', ec='#222', lw=0.5, label='NCCL/RDMA'),
    Patch(fc='#5A8DB5', ec='#222', lw=0.5, label='DPA (ours)'),
    mlines.Line2D([], [], color='#8B1A1A', ls='--', lw=2, label='GDR 4ch'),
]
fig.legend(handles=legend_h, loc='upper center', ncol=5, frameon=True,
           edgecolor='#CCC', fancybox=False, bbox_to_anchor=(0.5, 1.01), fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("throughput_comparison.pdf", format='pdf', bbox_inches='tight')
plt.show()