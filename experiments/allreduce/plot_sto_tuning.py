import matplotlib.pyplot as plt

import tuning_data as data

def get_timeout_ratio(exp, all=None):
  pts = []
  for k,v in exp.items():
      if "to" in v and v.get("to"):
          pts.append( (float(k), sum(v["to"]) / len(v["to"]) / ( all if all is not None else v['all'])))
  return [x for x,_ in pts], [y for _,y in pts]

def get_timeout_count(exp, op='avg', all=None):
  pts = []
  if op not in ['avg','min','max']: raise "invalid op"
  for k,v in exp.items():
      if "to" in v and v.get("to"):
          v = (sum(v["to"]) / len(v["to"])) if op == 'avg' else min(v["to"]) if op == 'min' else max(v["to"])
          pts.append( (float(k), v) )
  return [x for x,_ in pts], [y for _,y in pts]

def get_series(data, key):
  pts = []
  for k,v in data.items():
      if key in v and v.get(key):
          pts.append( ( float(k), sum(v[key]) / len(v[key]) ))
  return [x for x,_ in pts], [y for _,y in pts]


def get_peak(data, key, x0=0, x1=1):
  maximum = 0
  for k,v in data.items():
      if key in v and v.get(key):
          maximum = max([maximum, max(v[key])])
  return [x0,x1],[maximum,maximum]

# natural_sync = data.packets["natural-nosa-nore-sync"]
# natural_async = data["natural-nosa-nore-async"]

# fig, axl = plt.subplots()
# rows, cols = 2, 3
# fig, axs = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.2 * rows), constrained_layout=True)

#=================
# Plot 1
#=================
# axl, axr = axs[0,0], axs[0,0].twinx()
# axl.set_xlim(0.25, 15.0)
# xticks=[0.25 * i for i in range(1,60)]
# axl.set_xticks(xticks)
# axl.set_xticklabels([f"{v:g}" if i % 4 == 0 else "" for i, v in enumerate(xticks, 1)])
# axl.set_ylim(0,70)
# axl.set_yticks(list(range(0, 66, 5)))
# axl.set_yticklabels([str(t) if t % 10 == 0 else "" for t in list(range(0, 66, 5))])
# axl.set_xlabel("Straggler threshold (ms)", fontweight='bold')
# axl.set_ylabel("Throughput (Gb/s)", fontweight='bold')
# axr.set_ylim(0,0.01) # 1%
# axr.set_ylabel("Straggler timeouts (to_pkts / all_pkts)", fontweight='bold')

# # right axis, timeouts
# axr.plot(*get_timeout_ratio(data["natural-nosa-nore-sync"]), marker="o", color='lightgray', label="natural-nore-sync (right)")
# axr.plot(*get_timeout_ratio(data["natural-nosa-nore-async"]), marker="o", color="gray", label="natural-nore-async (right)")
# axr.plot(*get_timeout_ratio(data["natural-nosa-nore-async-warm-4"]), marker="x", color="black", label="natural-nore-async-warm (right)")
# # peak throughput
# axl.plot(*get_peak(data["natural-nosa-nore-async"],"gbit", 0.25, 15), 
#          linestyle="--", marker="", linewidth=3, markersize=16, color="red", label="peak (no retransmissions)")
# # timeout strategies, natural + 1 induced straggler
# axl.plot(*get_series(data["natural-sa-0.15-async"], "gbit"), marker="*", color="green", label="natural-sa-0.15-async")
# axl.plot(*get_series(data["natural-sa-2nd-async"], "gbit"), marker="x", color="green", label="natural-sa-2nd-async")
# axl.plot(*get_series(data["strag.1-sa-0.15-async"], "gbit"), marker="*", color="purple", label="strag.1-sa-0.15-async")
# axl.plot(*get_series(data["strag.1-sa-2nd-async"], "gbit"), marker="x", color="purple", label="strag.1-sa-2nd-async")

# axl.add_artist(axl.legend(loc="lower left", bbox_to_anchor=(-0.002, 0.05)))
# axr.legend(loc="lower right", bbox_to_anchor=(1.0, 0.05))



# plt.legend()
# plt.tight_layout()

rows, cols = 2, 3
fig, axs = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.2 * rows), constrained_layout=True)

#=================
# Plot 1
#=================

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

ax = axs[0,0]
ax.set_title("Finding straggler timeout (switch-side)", fontweight="bold")
ax.set_xticks(range(0,11), minor=True)
ax.set_xlim(-0.05, 7)
ax.set_ylim(0,3500)
ax.set_ylabel("Straggler timeouts (pkts)", fontweight='bold')
ax.set_xlabel("Straggler threshold (ms)", fontweight='bold')

ax.plot(*get_timeout_count(data.packets["natural-su-nowarm-batch20"], 'min'), marker="^", markersize=10, markerfacecolor='none', color="red", label="su-min")
ax.plot(*get_timeout_count(data.packets["natural-su-nowarm-batch20"], 'max'), marker="v", markersize=10, markerfacecolor='none', color="blue", label="su-max")
ax.plot(*get_timeout_count(data.packets["natural-su-nowarm-batch20"], 'avg'), marker="o", markersize=10, markerfacecolor='none', color="green", label="su-avg")

any0, maj0, all0 = find_zeroes(data.packets["natural-su-nowarm-batch20"])

vline(ax, any0, color="red",   linestyle=":", label="first")
vline(ax, maj0, color="black", linestyle=":", label="majority")
vline(ax, all0, color="black", linestyle="-", label="all")

ax.legend(loc="upper right", bbox_to_anchor=(0.85, 0.55))

total_pkts = list(data.packets["natural-su-nowarm-batch20"].values())[0]["all"]
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(ax.get_yticks())
ax2.set_yticklabels([f"{(y/total_pkts)*100:.3f}" for y in ax.get_yticks()])
ax2.set_ylabel("Percentage", fontweight='bold', fontsize='12')

# 3. FIX THE LEGEND: Get handles from ax, draw on ax2 (the top layer)
handles, labels = ax.get_legend_handles_labels()



# y1_min, y1_max = ax.get_ylim()
# ax2.set_ylim((y1_min / total_pkts) * 100, (y1_max / total_pkts) * 100)







# ax.plot(*get_timeout_count(data["natural-nosa-nore-sync"]), marker="None", color='lightgray', label="natural-nore-sync")
# ax.plot(*get_timeout_count(data["natural-nosa-nore-async"]), linestyle='None', marker="o", markerfacecolor='none', markersize=10, color="gray", label="natural-nore-async-nowarm")
# ax.plot(*get_timeout_count(data["natural-nosa-nore-async-warm-4"]), linestyle='None', marker="x", color="blue", label="natural-nore-async-warm-4")
# ax.plot(*get_timeout_count(data["natural-nosa-nore-async-warm-10"]), linestyle='None', marker="s", markerfacecolor='none', markersize=10, color="green", label="natural-nore-async-warm-10")
# ax.plot(*get_timeout_count(data["natural-nosa-nore-async-warm-20"]), linestyle='None', marker="^", markerfacecolor='none', markersize=10, color="red", label="natural-nore-async-warm-20")


plt.show()
# import matplotlib.pyplot as plt

# def mean(x): return sum(x)/len(x)

# def series(group, mode, f):
#     xs = sorted(float(k) for k in group)
#     ys = [(x, f(group[f"{x:g}"].get(mode, {}))) for x in xs]
#     ys = [(x,y) for x,y in ys if y is not None]
#     return [x for x,_ in ys], [y for _,y in ys]

# to_avg   = lambda d: None if not d.get("to")   else mean(d["to"])
# gbit_avg = lambda d: None if not d.get("gbit") else mean(d["gbit"])

# s0, s1 = data["straggle-0"], data["straggle-1"]

# fig, axL = plt.subplots()
# axR = axL.twinx()

# for m in ("1", "20"):
#     xT, T = series(s0, m, to_avg)
#     if T:
#         axL.plot(xT, [t/max(T) for t in T], marker="o", label=f"s0 {m} timeouts/max")

#     xG, G = series(s1, m, gbit_avg)
#     if G:
#         axR.plot(xG, [g/max(G) for g in G], marker="s", linestyle="--", label=f"s1 {m} gbit/max")

# axL.set_xlabel("sto")
# axL.set_ylabel("timeouts (normalized, 0 = 0 timeouts)")
# axR.set_ylabel("gbit (normalized)")

# axL.set_ylim(0, 1)
# axR.set_ylim(0, 1)

# xt = sorted(set().union(*[ln.get_xdata() for ln in axL.get_lines() + axR.get_lines()]))
# axL.set_xticks(xt)
# axL.set_xticklabels([f"{v:g}" for v in xt])

# h1,l1 = axL.get_legend_handles_labels()
# h2,l2 = axR.get_legend_handles_labels()
# axL.legend(h1+h2, l1+l2, loc="best")

# plt.tight_layout()
# plt.show()
