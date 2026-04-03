import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch

# ==========================================================
# DATA
# ==========================================================
with open(Path(__file__).parent / "bench.json") as f:
    tp = json.load(f)["allreduce-gpu"]["throughput"]

def avg_gbit(runs):
    if not runs: return None, 0.0
    g = [r["gbit_per_sec"] for r in runs]
    return np.mean(g), (np.std(g) if len(g) > 1 else 0.0)

# Primary bars (full width) — order matters
PRIMARY = [
    "dpa-256", "dpa-128", "dpa-64",
    "gloo",
    "nccl-tcp-tree", "nccl-tcp-ring",
    "nccl-rdma-tree", "nccl-rdma-tree-gdr/1",
    "nccl-rdma-ring", "nccl-rdma-ring-gdr/1",
    
]
# Ceiling markers: base_key -> [(overlay_key, linestyle, label), ...]
OVERLAYS = {
    "nccl-rdma-tree-gdr/1": [("nccl-rdma-tree-gdr/4", "--")],
    "nccl-rdma-ring-gdr/1": [("nccl-rdma-ring-gdr/4", "--")],
}

# ==========================================================
# COLORS — grouped
# ==========================================================
def color_of(k):
    if k == "gloo":              return "#8C7E78"
    if k.startswith("nccl-tcp"): return "#C4843E" if "tree" in k else "#D4A261"
    if k.startswith("nccl-rdma"):
        base = {"tree": "#5B8373", "ring": "#2E5645"}
        return base["ring"] if "ring" in k else base["tree"]
    if k.startswith("dpa"):
        return {"dpa-64": "#8BADC7", "dpa-128": "#5A8DB5", "dpa-256": "#2D6A9F"}.get(k, "#5A8DB5")
    return "#888"

# ==========================================================
# PLOT
# ==========================================================
plt.rcParams.update({'font.size': 8, 'font.family': 'serif'})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.4), dpi=150)

W_MAIN = 0.72

global_max = 0  # track for shared y-axis

for size, ax, title in [("25", ax1, "25 M elements (100 MB)"), ("100", ax2, "100 M elements (400 MB)")]:
    # collect primary bars that have data at this size
    keys, vals, errs = [], [], []
    for k in PRIMARY:
        runs = tp.get(k, {}).get("data", {}).get(size, [])
        if not runs: continue
        m, s = avg_gbit(runs)
        keys.append(k); vals.append(m); errs.append(s)

    x = np.arange(len(keys))
    colors = [color_of(k) for k in keys]

    bars = ax.bar(x, vals, W_MAIN, yerr=errs, capsize=2,
                  color=colors, edgecolor='#222', linewidth=0.5,
                  error_kw=dict(lw=0.7, capthick=0.5), zorder=3)

    # value labels
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + errs[i] + 0.6,
                f'{v:.1f}', ha='center', va='bottom', fontsize=5.5, fontweight='bold', color='#333')

    # ceiling markers for GDR channel variants
    for i, k in enumerate(keys):
        if k not in OVERLAYS: continue
        bx = x[i] - W_MAIN/2
        for ovl_key, ls in OVERLAYS[k]:
            runs = tp.get(ovl_key, {}).get("data", {}).get(size, [])
            if not runs: continue
            m, _ = avg_gbit(runs)
            ax.hlines(m, bx, bx + W_MAIN, linestyles=ls, colors='#8B1A1A',
                      linewidth=2.0, zorder=5)

    ax.set_title(title, fontweight='bold', fontsize=9, pad=8)
    ax.set_ylabel("Throughput (Gbit/s)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=40, ha='right', fontsize=6.5)
    ax.set_xlim(-0.6, len(keys) - 0.3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#DDD', linewidth=0.5, zorder=0)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    ymax = max(vals) * 1.15
    if ymax > global_max: global_max = ymax

# shared y scale
for a in (ax1, ax2):
    a.set_ylim(0, global_max)

# legend
import matplotlib.lines as mlines
legend_h = [
    Patch(fc='#8C7E78', ec='#222', lw=0.5, label='Gloo'),
    Patch(fc='#C4843E', ec='#222', lw=0.5, label='NCCL/TCP'),
    Patch(fc='#5B8373', ec='#222', lw=0.5, label='NCCL/RDMA'),
    Patch(fc='#5A8DB5', ec='#222', lw=0.5, label='DPA (ours)'),
    mlines.Line2D([], [], color='#8B1A1A', ls='--', lw=2.0, label='GDR 4ch'),
]
fig.legend(handles=legend_h, loc='upper center', ncol=5, frameon=True,
           edgecolor='#CCC', fancybox=False, bbox_to_anchor=(0.5, 1.02), fontsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("throughput_comparison.pdf", format='pdf', bbox_inches='tight')
plt.show()