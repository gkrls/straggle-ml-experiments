data = {
  "straggle-0" : {
    "0.25": {
      "tsc"  : 4,
      "real" : 0.262,
      "single"  :{"ok":[586646,587286,587247],      "to":[1600,960,999],  "gbit":[23.83,23.78,23.75],"time":[33.57,33.63,33.67]},
      "batch-10":{"ok":[5880540,5880540,5880540],   "to":[1920,1920,1920],"gbit":[53.45,53.56,53.73],"time":[14.96,14.93]},
      "batch-20":{"ok":[11762936,11762923,11762616],"to":[1984,1997,2304],"gbit":[60.75,60.58,61.28],"time":[13.16,13.20,13.05]},
      "sync-10" :{"ok":[5881308,5881173,5881183],   "to":[1152,1287,1277],"gbit":[25.62,25.63,25.59],"time":[31.22,31.21,31.25]},
      "sync-20" :{"ok":[11763939,11763116,11762893],"to":[981,1804,2027], "gbit":[25.58,25.63,25.48],"time":[31.26,31.20,31.39]},
    },
    "0.5": {
      "tsc" : 8,
      "real": 0.524,
      "single"  : {"ok":[587734,587414,586982],      "to":[512,832,1264],  "gbit":[23.77,24.19,23.90],"time":[33.64,33.06,33.46]},
      "batch-20": {"ok":[11762680,11761784,11762232],"to":[2240,3136,2688],"gbit":[60.92,60.62,60.82],"time":[13.12,13.19,13.15]},
      "sync-20" : {"ok":[11764162,11764626,11764276],"to":[758,294,644],   "gbit":[25.63,25.74,25.95],"time":[31.22,31.07,30.82]},
    },
    "1.0": {
      "single"  : {"ok":[587734,587414,586982],      "to":[512,832,1264],  "gbit":[23.77,24.19,23.90],"time":[33.64,33.06,33.46]},
    }
  },
  "straggle-1": {
    "0.25": {
      "tsc" : 4,
      "real": 0.262,
      "single"  : {"gbit":[23.95,24.00,24.18],"time":[33.39,33.33,33.08]},
      "batch-10": {"gbit":[53.26,52.74,52.59],"time":[15.02,15.16,15.21]},
      "batch-20": {"gbit":[60.40,60.76,60.69],"time":[13.24,13.16,13.18]},
      "synch-20": {"gbit":[24.05,23.63,23.53],"time":[33.25,33.84,33.99]},
    },
    "0.5": {
      "tsc" : 8,
      "real": 0.524,
      "single"  : {},
      "batch-20": {"gbit":[59.12,58.48,58.13],"time":[13.52,13.67,13.76]},
      "sync-20" : {}
    },
  }
}

import matplotlib.pyplot as plt

def mean(x): return sum(x)/len(x)

def ok_rate(d):
    ok,to = d.get("ok"), d.get("to")
    return None if not (ok and to) else mean([o/(o+t) for o,t in zip(ok,to)])

def avg_gbit(d):
    g = d.get("gbit")
    return None if not g else mean(g)

def series(group, mode, f):
    xs = sorted(float(k) for k in group)
    ys = [(x, f(group[f"{x:g}"].get(mode, {}))) for x in xs]
    ys = [(x,y) for x,y in ys if y is not None]
    return [x for x,_ in ys], [y for _,y in ys]

def zoom(ax, ys, pad=0.15):
    lo, hi = min(ys), max(ys)
    if hi == lo: ax.set_ylim(lo-1, hi+1); return
    m = (hi - lo) * pad
    ax.set_ylim(lo - m, hi + m)

s0, s1 = data["straggle-0"], data["straggle-1"]

fig, axL = plt.subplots()
axR = axL.twinx()

Lall, Rall = [], []

# left: straggle-0 success ratio
for m in ("single", "batch-20"):
    x, y = series(s0, m, ok_rate)
    axL.plot(x, y, marker="o", label=f"s0 {m} success")
    Lall += y

# right: straggle-1 gbit
for m in ("single", "batch-20"):
    x, y = series(s1, m, avg_gbit)
    if y:
        axR.plot(x, y, marker="s", linestyle="--", label=f"s1 {m} gbit")
        Rall += y

axL.set_xlabel("sto")
axL.set_ylabel("success ratio (ok/(ok+to))")
axR.set_ylabel("throughput (gbit/s)")

zoom(axL, Lall)
zoom(axR, Rall)

xt = sorted(set().union(*[ln.get_xdata() for ln in axL.get_lines() + axR.get_lines()]))
axL.set_xticks(xt); axL.set_xticklabels([f"{v:g}" for v in xt])

h1,l1 = axL.get_legend_handles_labels()
h2,l2 = axR.get_legend_handles_labels()
axL.legend(h1+h2, l1+l2, loc="best")

plt.tight_layout()
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

# for m in ("single", "batch-20"):
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
