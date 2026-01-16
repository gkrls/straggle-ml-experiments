data = {
  "straggle-0" : {
    "0.25": {
      "tsc": 4, "real": 0.262,
      "sync-20" :{"ok":[11763939,11763116,11762893],"to":[981,1804,2027], "gbit":[25.58,25.63,25.48],"time":[31.26,31.20,31.39]},
      "batch-1" :{"ok":[586646,587286,587247],      "to":[1600,960,999],  "gbit":[23.83,23.78,23.75],"time":[33.57,33.63,33.67]},
      "batch-20":{"ok":[11762936,11762923,11762616],"to":[1984,1997,2304],"gbit":[60.75,60.58,61.28],"time":[13.16,13.20,13.05]},
    },
    "0.5": {
      "tsc" : 8, "real": 0.524,
      "sync-20" :{"ok":[11764162,11764626,11764276],"to":[758,294,644],   "gbit":[25.63,25.74,25.95],"time":[31.22,31.07,30.82]},
      "batch-1" :{"ok":[587734,587414,586982],      "to":[512,832,1264],  "gbit":[23.77,24.19,23.90],"time":[33.64,33.06,33.46]},
      "batch-20":{"ok":[11762680,11761784,11762232],"to":[2240,3136,2688],"gbit":[60.92,60.62,60.82],"time":[13.12,13.19,13.15]},
    },
    "1.0": {
      "tsc": 16, "real": 1.048,
      "batch-1" :{"ok":[588246,588246,588246],      "to":[0,0,0],      },
      "batch-20":{"ok":[11762360,11764144,11764536],"to":[2560,776,384]}
    },
    "1.25": {
      "tsc": 20,
      "real": 1.31,
      "batch-1" :{"ok":[588246,588246,588246],      "to":[0,0,0]},
      "batch-20":{"ok":[11763000,11762360,11763000],"to":[1920,2560,1920]}
    }
  },
  "straggle-1": {
    "0.25": {"tsc":4, "real":0.262},
    "0.50": {"tsc":8, "real":0.524, "rto": 0.26, "gbit":[59.12,58.48,58.13], "time":[13.52,13.67,13.76] },
    "1.00": {"gbit":[55.32],"time":[14.45]},
    "1.25": {"gbit":[57.40,57.63,56.91,57.39],"time":[13.93,13.87,14.05,13.93]},
    "1.50": {},
    "2.00": {},
    "3.00": {},
    "5.00": {},
    "7.50": {},
    "10.0": {},
    "12.5": {}
  },
  "straggle-1-static-rto" : {
    "1.25": {"tsc" : 20, "real": 1.31, "gbit": {}}
  },
  "straggle-1-sync": {
    "0.25": {"tsc":4, "real":0.262, "rto":0.14, "gbit":[53.69,55.03,53.84,55.55], "time":[14.89,14.53,14.85,14.40]},
    "0.50": {"tsc":8, "real":0.524, "rto":0.27, "gbit":[56.60,56.75,56.17,56.58], "time":[14.15,14.09,14.24,14.13]},
    "0.75": {"tsc":12,"real":0.786, "rto":0.40, "gbit":[57.96,57.29,56.50,57.67], "time":[13.80,13.96,14.15,13.86]},
    "1.00": {"tsc":16,"real":1.048, "rto":0.53, "gbit":[56.88,55.44,56.22,55.99], "time":[14.06,14.42,14.22,14.28]},
    "1.25": {"tsc":20,"real":1.310, "rto":0.66, "gbit":[], "time": []},
  },
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
    items = sorted(((float(k), v) for k, v in group.items()), key=lambda t: t[0])
    pts = [(x, f(v.get(mode, {}))) for x, v in items]
    pts = [(x,y) for x,y in pts if y is not None]
    return [x for x,_ in pts], [y for _,y in pts]

def zoom(ax, ys, pad=0.15):
    lo, hi = min(ys), max(ys)
    m = (hi - lo) * pad if hi != lo else 1
    ax.set_ylim(lo - m, hi + m)

s0 = data["straggle-0"]
s1 = data["straggle-1"]

fig, axL = plt.subplots()
axR = axL.twinx()

# Left: straggle-0 success (batch-1 + batch-20) -> different default colors
Lall = []
for mode in ("batch-1", "batch-20"):
    x, y = series(s0, mode, ok_rate)
    axL.plot(x, y, marker="o", label=f"s0 {mode} success")
    Lall += y

# Right: straggle-1 throughput (batch-20 only)
xR, yR = series(s1, "batch-20", avg_gbit)
axR.plot(xR, yR, marker="s", linestyle="--", label="s1 batch-20 gbit")

axL.set_xlabel("sto")
axL.set_ylabel("success ratio (ok/(ok+to))")
axR.set_ylabel("throughput (gbit/s)")

zoom(axL, Lall)
zoom(axR, yR)

xt = sorted(set().union(*[ln.get_xdata() for ln in axL.get_lines() + axR.get_lines()]))
axL.set_xticks(xt)
axL.set_xticklabels([f"{v:g}" for v in xt])

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

# for m in ("batch-1", "batch-20"):
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
