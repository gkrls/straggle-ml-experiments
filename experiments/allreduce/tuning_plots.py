import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

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

fig = plt.figure(figsize=(14, 8),constrained_layout=True)

gs = fig.add_gridspec(2, 1)
rows = [gs[0, 0].subgridspec(1, 3, width_ratios=[2, 0.5, 2]), 
        gs[1, 0].subgridspec(1, 3)]
axs = [[fig.add_subplot(rows[r][0, c]) for c in range(rows[r].ncols)] for r in range(len(rows))]


#=========================================
# Plot 0,0 - switch-side straggler timeout
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
  return any0, maj0, all0

def vline(ax, x, label, color, linestyle):
    ax.axvline(x, color=color, linestyle=linestyle)
    ax.text(x + 0.03, ax.get_ylim()[1] * 0.95, label, fontsize=13, rotation=90, color=color, va="top", ha="left")

ax = axs[0][0]#fig.add_subplot(row0[0,0])
ax.set_title("Profiling straggler timeout (switch-side)", fontweight="bold")
ax.set_xticks(range(0,11), minor=True)
ax.set_xlim(-0.05, 7)
ax.set_ylim(0,3500)
ax.set_ylabel("Straggler timeouts (pkts)", fontweight='bold')
ax.set_xlabel("Straggler threshold (ms)", fontweight='bold')

ax.plot(*get_timeouts(data.packets["natural"]["su-nore"], 'min'), marker="^", markersize=10, markerfacecolor='none', color="red", label="su-min")
ax.plot(*get_timeouts(data.packets["natural"]["su-nore"], 'max'), marker="v", markersize=10, markerfacecolor='none', color="blue", label="su-max")
ax.plot(*get_timeouts(data.packets["natural"]["su-nore"], 'avg'), marker="o", markersize=10, markerfacecolor='none', color="green", label="su-avg")

any0, maj0, all0 = find_zeroes(data.packets["natural"]["su-nore"])

vline(ax, any0, color="red",   linestyle=":", label="first")
vline(ax, maj0, color="black", linestyle=":", label="majority")
vline(ax, all0, color="black", linestyle="-", label="all")

ax.legend(loc="upper right", bbox_to_anchor=(0.85, 0.55))

total_pkts = list(data.packets["natural"]["su-nore"].values())[0]["all"]
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(ax.get_yticks())
ax2.set_yticklabels([f"{(y/total_pkts)*100:.3f}" for y in ax.get_yticks()])
ax2.set_ylabel("Percentage", fontweight='bold', fontsize='12')



# ax.plot(*get_timeout_count(data["natural-nosa-nore-sync"]), marker="None", color='lightgray', label="natural-nore-sync")
# ax.plot(*get_timeout_count(data["natural-nosa-nore-async"]), linestyle='None', marker="o", markerfacecolor='none', markersize=10, color="gray", label="natural-nore-async-nowarm")
# ax.plot(*get_timeout_count(data["natural-nosa-nore-async-warm-4"]), linestyle='None', marker="x", color="blue", label="natural-nore-async-warm-4")
# ax.plot(*get_timeout_count(data["natural-nosa-nore-async-warm-10"]), linestyle='None', marker="s", markerfacecolor='none', markersize=10, color="green", label="natural-nore-async-warm-10")
# ax.plot(*get_timeout_count(data["natural-nosa-nore-async-warm-20"]), linestyle='None', marker="^", markerfacecolor='none', markersize=10, color="red", label="natural-nore-async-warm-20")


#=========================================
# Plot 0,1 - peak throughput
#=========================================
ax = axs[0][1] #fig.add_subplot(row0[0,1])


save_subplot(fig, axs[0][0], "packets.pdf")

plt.show()
