import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.patches import Patch

import tuning_data as data


def save_subplot(fig, ax, filename="plot.pdf"):
    fig.canvas.draw()
    relevant_axes = [a for a in ax.figure.axes if a.get_subplotspec() == ax.get_subplotspec()]
    renderer = ax.figure.canvas.get_renderer()
    bboxes = [a.get_tightbbox(renderer) for a in relevant_axes]
    full_bbox = Bbox.union(bboxes)
    extent = full_bbox.transformed(ax.figure.dpi_scale_trans.inverted())
    ax.figure.savefig(filename, bbox_inches=extent)

# # 2. CALL THIS BEFORE plt.show()
# save_subplot_no_bleed(ax, ax2, "straggler_timeout.pdf")

# rows, cols = 2, 3
# fig, axs = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.2 * rows), constrained_layout=True)

RTT = 40

BDP = {
    40 : 500000,
    45 : 562500,
    50 : 625000
}

# 66 + 32 + payload * 4
PACKET_SIZE_344   = 1474 #  86 values/pipe, 43 reducers/pipe, 11 reducer stages, tofino 2 only (frame-limit)
PACKET_SIZE_256_Q = 1122 #  64 values/pipe, 32 reducers/pipe,  8 reducer stages, tofino 1+2
# dual_pipe
PACKET_SIZE_256_D = 1122 # 128 values/pipe, 64 reducers/pipe, 16 reducer states, tofino 2 only (pipe-limit)
PACKET_SIZE_128_D = 610  #  64 values/pipe, 32 reducers/pipe,  8 reducer stages, tofino 1+2
# single_pipe
PACKET_SIZE_128_S = 610  # 128 values/pipe, 64 reducers/pipe, 16 reducer stages, tofino 2 only (pipe-limit)
PACKET_SIZE_86    = 442  #  86 values/pipe, 43 reducers/pipe, 11 reducer stages, tofino 2 only (frame-limit)
PACKET_SIZE_64    = 354  #  64 values/pipe, 32 reducers/pipe,  8 reducer stages, tofino 1+2

fig = plt.figure(figsize=(14, 8),constrained_layout=True)

gs = fig.add_gridspec(2, 1)
rows = [gs[0, 0].subgridspec(1, 3), # width_ratios=[2, 0.5, 2] 
        gs[1, 0].subgridspec(1, 3)]
axs = [[fig.add_subplot(rows[r][0, c]) for c in range(rows[r].ncols)] for r in range(len(rows))]

#=========================================
# Plot 0,0 - window size
#=========================================
ax = axs[0][0]
ax2 = ax.twinx()

window_data = data.window["4-pipe"]

# 1. Setup window sizes and thread counts
window_sizes = list(window_data.keys())
# Get threads from the first window entry that actually has data
first_window_key = window_sizes[0]
threads = list(window_data[first_window_key].keys())

# 2. Prepare data structures
means = {t: [] for t in threads}
mn_err = {t: [] for t in threads}
mx_err = {t: [] for t in threads}

time_means = {t: [] for t in threads}

for w in window_sizes:
    for t in threads:
        gbit = window_data.get(w, {}).get(t, {}).get("gbit", [0])
        time = window_data.get(w, {}).get(t, {}).get("time", [0])

        if gbit:
            avg = sum(gbit) / len(gbit)
            means[t].append(avg)
            mn_err[t].append(avg - min(gbit))
            mx_err[t].append(max(gbit) - avg)
        else:
            means[t].append(0)
            mn_err[t].append(0)
            mx_err[t].append(0)
        if time:
            time_means[t].append(sum(time) / len(time))
        else:
            time_means[t].append(0)

x_indices = list(range(len(window_sizes)))
width = 0.2
num_threads = len(threads)
colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']
for i, t in enumerate(threads):
    offset = (i - (num_threads - 1) / 2) * width # offset to center the cluster of bars
    pos = [x + offset for x in x_indices]
    yerr = [mn_err[t], mx_err[t]] # Matplotlib expects error bars as [ [lower_offsets], [upper_offsets] ]
    ax.bar(pos, means[t], width, label=f"{t}T", yerr=yerr, capsize=4, alpha=0.8)
    ax2.bar(pos,time_means[t],width,facecolor='none',edgecolor='black',linewidth=1.5,hatch='//',label=f"ms")
 
ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.3)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0,72)
ax.set_ylabel('Throughput (Gbit/s)',fontweight='bold')
ax.set_xlabel('Window size (packets)',fontweight='bold')
ax.set_title(f"Profiling window size (BDP@40={BDP[40] / 1000:.1f}K,frame={PACKET_SIZE_256_Q})",fontweight='bold',fontsize=10)
ax.set_xticks(x_indices)
ax.set_xticklabels(window_sizes)


ax2.set_ylim(0,105)   
ax2.set_ylabel("Amortized per-op latency (ms)",fontweight='bold',labelpad=0)

handles, labels = ax.get_legend_handles_labels()
handles.append(Patch(facecolor='white', edgecolor='black', hatch='//', label='Latency'))
ax2.legend(handles=handles,loc='upper left',bbox_to_anchor=(0,1),ncol=len(handles),
            columnspacing=0.5,handletextpad=0.4,frameon=True)


#=========================================
# Plot 0,1 - switch-side straggler timeout
#=========================================
def get_timeouts(exp, op='avg', all=None):
  pts = []
  if op not in ['avg','min','max']: raise "invalid op"
  for k,v in exp.items():
      if "to" in v and v.get("to"):
          v = (sum(v["to"]) / len(v["to"])) if op == 'avg' else min(v["to"]) if op == 'min' else max(v["to"])
          pts.append( (float(k), v) )
  return [x for x,_ in pts], [y for _,y in pts]

def find_zeroes(data):
  any0,maj0,all0 = None, None, None
  xs = sorted(float(k) for k in data.keys())
  for x in xs:
      arr = data[f"{x:.2f}"]["to"]
      zeros = sum(v == 0 for v in arr)
      if any0 is None and zeros >= 1: any0 = x
      if maj0 is None and zeros >= len(arr) / 2: maj0 = x
      if all0 is None and zeros == len(arr): all0 = x
  return any0 if any0 is not None else 0, maj0 if maj0 is not None else 0, all0 if all0 is not None else 0,

def vline(ax, x, label, color, linestyle):
    ax.axvline(x, color=color, linestyle=linestyle)
    ax.text(x + 0.03, ax.get_ylim()[1] * 0.95, label, fontsize=13, rotation=90, color=color, va="top", ha="left")

ax = axs[0][1]#fig.add_subplot(row0[0,0])
ax.set_title("Profiling straggler timeout (win=384/6)", fontweight="bold", fontsize=10)
ax.set_xticks(range(0,11), minor=True)
ax.set_xlim(-0.05, 7)
ax.set_ylim(0,3500)
ax.set_ylabel("Straggler timeouts (pkts)", fontweight='bold')
ax.set_xlabel("Straggler threshold (ms)", fontweight='bold')

ax.plot(*get_timeouts(data.packets["384"]["natural-su-nore"], 'min'), marker="^", markersize=10, markerfacecolor='none', color="red", label="su-min")
ax.plot(*get_timeouts(data.packets["384"]["natural-su-nore"], 'max'), marker="v", markersize=10, markerfacecolor='none', color="blue", label="su-max")
ax.plot(*get_timeouts(data.packets["384"]["natural-su-nore"], 'avg'), marker="o", markersize=10, markerfacecolor='none', color="green", label="su-avg")

any0, maj0, all0 = find_zeroes(data.packets["384"]["natural-su-nore"])

vline(ax, any0, color="red",   linestyle=":", label="first")
vline(ax, maj0, color="black", linestyle=":", label="majority")
vline(ax, all0, color="black", linestyle="-", label="all")

ax.legend(loc="upper right", bbox_to_anchor=(0.85, 0.55))

total_pkts = list(data.packets["384"]["natural-su-nore"].values())[0]["all"]
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(ax.get_yticks())
ax2.set_yticklabels([f"{(y/total_pkts)*100:.3f}" for y in ax.get_yticks()])
ax2.set_ylabel("Percentage", fontweight='bold')



# ax.plot(*get_timeout_count(data["natural-nosa-nore-sync"]), marker="None", color='lightgray', label="natural-nore-sync")
# ax.plot(*get_timeout_count(data["natural-nosa-nore-async"]), linestyle='None', marker="o", markerfacecolor='none', markersize=10, color="gray", label="natural-nore-async-nowarm")
# ax.plot(*get_timeout_count(data["natural-nosa-nore-async-warm-4"]), linestyle='None', marker="x", color="blue", label="natural-nore-async-warm-4")
# ax.plot(*get_timeout_count(data["natural-nosa-nore-async-warm-10"]), linestyle='None', marker="s", markerfacecolor='none', markersize=10, color="green", label="natural-nore-async-warm-10")
# ax.plot(*get_timeout_count(data["natural-nosa-nore-async-warm-20"]), linestyle='None', marker="^", markerfacecolor='none', markersize=10, color="red", label="natural-nore-async-warm-20")


#=========================================
# Plot 0,2 - peak throughput
#=========================================
ax = axs[0][2] #fig.add_subplot(row0[0,1])

ax.set_title("Profiling straggler timeout (win=446/6)", fontweight="bold", fontsize=10)

ax.plot(*get_timeouts(data.packets["446"]["natural-su-nore"], 'min'), marker="^", markersize=10, markerfacecolor='none', color="red", label="su-min")
ax.plot(*get_timeouts(data.packets["446"]["natural-su-nore"], 'max'), marker="v", markersize=10, markerfacecolor='none', color="blue", label="su-max")
ax.plot(*get_timeouts(data.packets["446"]["natural-su-nore"], 'avg'), marker="o", markersize=10, markerfacecolor='none', color="green", label="su-avg")

any0, maj0, all0 = find_zeroes(data.packets["446"]["natural-su-nore"])
vline(ax, any0, color="red",   linestyle=":", label="first")
vline(ax, maj0, color="black", linestyle=":", label="majority")
vline(ax, all0, color="black", linestyle="-", label="all")

ax.legend(loc="upper right", bbox_to_anchor=(0.85, 0.55))

total_pkts = list(data.packets["446"]["natural-su-nore"].values())[0]["all"]
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(ax.get_yticks())
ax2.set_yticklabels([f"{(y/total_pkts)*100:.3f}" for y in ax.get_yticks()])
ax2.set_ylabel("Percentage", fontweight='bold')

#=========================================
#=========================================
#=========================================
#=========================================
#=========================================
#=========================================

save_subplot(fig, axs[0][0], "window.pdf")
save_subplot(fig, axs[0][1], "packets.pdf")

plt.show()
