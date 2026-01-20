data = {
  "batch-natural-nosa-nore-sync": { # batch-20, no induced stragglers, sync threads, sa_disabled, re_disabled
    #"0.25": {"tsc":4,  "real":0.262, "rto":0.14,"all":11764920,"to":[4032,4224,3648,4032],"gbit":[56.96,55.03,55.02,55.55],"time":[14.04,14.53,14.53,14.40]},
    "0.50": {"tsc":8,  "real":0.524, "rto":0.27,"all":11764920,"to":[2944,4608,3072,4096],"gbit":[56.60,56.75,56.17,56.58],"time":[14.15,14.09,14.24,14.13]},
    #"0.75": {"tsc":12, "real":0.786, "rto":0.40,"all":11764920,"to":[3648,1920,1856,4608],"gbit":[57.96,57.29,56.50,57.67],"time":[13.80,13.96,14.15,13.86]},
    #"1.25": {"tsc":20, "real":1.310, "rto":0.66,"all":11764920,"to":[2304,3648,1920,3584],"gbit":[57.62,56.41,56.37,56.87],"time":[13.88,14.18,14.18,14.06]},


    # "1.00": {"tsc":16, "real":1.048, "rto":0.53,"all":11764920,"to":[2624,3174,3712,3520],"gbit":[56.88,55.44,56.22,55.99],"time":[14.06,14.42,14.22,14.28]},
   
    "1.00": {"tsc":16, "real":1.048, "rto":0.53,"all":11764920,"to":[2881,4160,1728,2944],"gbit":[56.10,54.66,56.88,56.13],"time":[14.26,14.63,14.06,14.25]},
    "1.50": {"tsc":23, "real":1.507, "rto":0.76,"all":11764920,"to":[2176,2752,3200,2763],"gbit":[55.11,55.49,54.14,54.75],"time":[14.51,14.41,14.77,14.61]},
    "2.00": {"tsc":31, "real":2.031, "rto":1.05,"all":11764920,"to":[2752,1344,3584,3008],"gbit":[54.85,57.44,56.30,56.26],"time":[14.58,13.92,14.20,14.21]},
    "2.50": {"tsc":39, "real":2.555, "rto":1.30,"all":11764920,"to":[1984,3456,2880,2304],"gbit":[56.09,56.88,55.60,57.64],"time":[14.26,14.06,14.38,13.87]},
    "3.00": {"tsc":46, "real":3.014, "rto":1.55,"all":11764920,"to":[2880,2528,1754,2816],"gbit":[56.44,55.41,56.05,57.49],"time":[14.17,14.43,14.27,13.91]},
    "3.50": {"tsc":54, "real":3.538, "rto":1.80,"all":11764920,"to":[1143, 704,1024, 576],"gbit":[56.85,56.28,56.48,56.94],"time":[14.07,14.21,14.16,14.04]},
    "4.00": {"tsc":62, "real":4.063, "rto":2.05,"all":11764920,"to":[   0, 384, 453,   0],"gbit":[56.11,54.75,56.88,54.92],"time":[14.25,14.61,14.06,14.56]},
    "5.00": {"tsc":77, "real":5.046, "rto":2.55,"all":11764920,"to":[   0,   0,   0,   0],"gbit":[58.20,58.25,59.23,57.01],"time":[13.74,13.73,13.50,14.03]},
    "6.00": {"tsc":92, "real":6.029, "rto":3.05,"all":11764920,"to":[ 128,   0,   0,   0],"gbit":[56.42,57.97,57.33,56.29],"time":[14.17,13.79,13.95,14.21]},
    "7.00": {"tsc":107,"real":7.012, "rto":3.55,"all":11764920,"to":[   0,   0,   0,   0],"gbit":[58.22,56.78,56.29,55.45],"time":[13.73,14.08,14.20,14.42]},
    "8.00": {"tsc":123,"real":8.060, "rto":4.05,"all":11764920,"to":[   0,   0,   0,   0],"gbit":[58.82,57.15,57.40,57.35],"time":[14.07,13.99,13.93,13.94]},    
    "9.00": {"tsc":138,"real":9.043, "rto":4.55,"all":11764920,"to":[   0,   0,   0,   0],"gbit":[56.94,55.71,56.77,55.96],"time":[14.04,14.35,14.09,14.29]},
    "10.0": {"tsc":153,"real":10.027,"rto":5.03,"all":11764920,"to":[   0,   0,   0,   0],"gbit":[55.66,56.96,56.16,57.97],"time":[14.37,14.04,14.24,13.80]},
    "12.5": {"tsc":191,"real":12.517,"rto":6.26,"all":11764920,"to":[   0,   0,   0,   0],"gbit":[56.04,57.44,57.22,57.35],"time":[14.27,13.92,13.97,13.94]},
    "15.0": {"tsc":229,"real":15.007,"rto":7.51,"all":11764920,"to":[   0,   0,   0,   0],"gbit":[56.37,57.12,56.15,54.99],"time":[14.19,14.00,14.24,14.54]},
  },

  "batch-natural-nosa-nore-async": {
    "0.25": {"tsc":4,  "real":0.262, "rto":0.14,"all":11764920,"to":[2304,2548,1915,3564],"gbit":[62.05,61.50,61.60,61.41],"time":[12.89,13.00,12.98,13.02]},
    "0.50": {"tsc":8,  "real":0.524, "rto":0.27,"all":11764920,"to":[1984,2624,1984,2593],"gbit":[60.15,60.69,61.06,59.68],"time":[13.29,13.18,13.10,13.40]},
    "1.00": {"tsc":16, "real":1.048, "rto":0.53,"all":11764920,"to":[1216,1920,2307,1536],"gbit":[60.37,60.36,61.27,60.90],"time":[13.24,13.25,13.05,13.13]},
    "1.50": {"tsc":23, "real":1.507, "rto":0.76,"all":11764920,"to":[ 870,1920,1563,1920],"gbit":[60.56,60.75,60.00,60.77],"time":[13.20,13.16,13.33,13.16]},
    "2.00": {"tsc":31, "real":2.031, "rto":1.05,"all":11764920,"to":[2240,1792, 704,1472],"gbit":[60.46,59.27,60.06,60.35],"time":[13.23,13.49,13.31,13.25]},
    "2.50": {"tsc":39, "real":2.555, "rto":1.30,"all":11764920,"to":[   2,1600,1728,1920],"gbit":[60.60,59.79,60.67,61.05],"time":[13.20,13.38,13.18,13.10]},
    "3.00": {"tsc":46, "real":3.014, "rto":1.55,"all":11764920,"to":[  64,1024, 975, 768],"gbit":[60.39,61.28,59.04,60.00],"time":[13.24,13.05,13.54,13.33]},
    "4.00": {"tsc":62, "real":4.063, "rto":2.05,"all":11764920,"to":[ 128,   0,   1, 640],"gbit":[60.84,61.13,60.40,59.39],"time":[13.14,13.08,13.24,13.46]},
    "5.00": {"tsc":77, "real":5.046, "rto":2.55,"all":11764920,"to":[   0,   0, 320,   0],"gbit":[60.85,61.11,61.57,60.29],"time":[13.14,13.01,12.99,13.26]},
    "6.00": {"tsc":92, "real":6.029, "rto":3.05,"all":11764920,"to":[   0,   0,   0,   0],"gbit":[60.40,61.14,61.08,61.71],"time":[13.24,13.34,13.09,12.96]},
    "7.00": {"tsc":107,"real":7.012, "rto":3.55,"all":11764920,"to":[   0,   0,   0,   0],"gbit":[60.75,61.18,60.82,60.94],"time":[13.16,13.07,13.15,13.12]},
    "8.00": {"tsc":123,"real":8.060, "rto":4.05,"all":11764920,"to":[   0,   0,   0,   0],"gbit":[60.84,61.17,61.28,59.88],"time":[13.14,13.07,13.05,13.35]},
    "9.00": {"tsc":138,"real":9.043, "rto":4.55,"all":11764920,"to":[   0,   0,   0,   0],"gbit":[60.26,59.65,59.88,61.14],"time":[13.27,13.40,13.35,13.08]},
    "10.0": {"tsc":153,"real":10.027,"rto":5.03,"all":11764920,"to":[   0,   0,   0,   0],"gbit":[60.57,61.02,60.73,59.87],"time":[13.20,13.10,13.17,13.36]},
    "12.5": {"tsc":191,"real":12.517,"rto":7.51,"all":11764920,"to":[   0,   0,   0,   0],"gbit":[60.37,59.93,60.78,60.24],"time":[13.25,13.34,13.16,13.27]},
    "15.0": {"tsc":229,"real":15.007,"rto":7.51,"all":11764920,"to":[   0,   0,   0,   0],"gbit":[59.45,60.69,60.54,60.60],"time":[13.45,13.18,13.21,13.20]}
  },

  "batch-natural-sa-re-async": {
      
  },

  "batch-straggle1-sa-stare-async": {
    "0.50": {},
  },

  "batch-straggle1-sa-dynre-async": {
    "0.50": {},
  }
}


import matplotlib.pyplot as plt

def mean(x): return sum(x) / len(x)

def series_success(exp):
    pts = []
    for k, v in exp.items():
        if "all" in v and v.get("to"):
            pts.append((float(k), (v["all"] - mean(v["to"])) / v["all"]))
    # pts.sort()
    return [x for x,_ in pts], [y for _,y in pts]

def series_timeout(exp):
  pts = []
  for k,v in exp.items():
      if "to" in v and v.get("to"):
          pts.append( (float(k), sum(v["to"]) / len(v["to"]) ))
  return [x for x,_ in pts], [y for _,y in pts]

def zoom(ax, ys, pad=0.15):
    lo, hi = min(ys), max(ys)
    m = (hi - lo) * pad if hi != lo else 1e-6
    ax.set_ylim(lo - m, hi + m)

natural_sync = data["batch-natural-nosa-nore-sync"]
natural_async = data["batch-natural-nosa-nore-async"]

fig, axl = plt.subplots()
xt = [0.25 * i for i in range(1, 61)]   # 0.25 .. 15.0
axl.set_xlim(0.25, 15.0)
axl.set_xticks(xt)
axl.set_xticklabels([f"{v:g}" if i % 4 == 0 else "" for i, v in enumerate(xt, 1)])
axr = axl.twinx()

# success packets sync
x,y = series_timeout(data["batch-natural-nosa-nore-sync"])
axl.plot(x, y, marker="o", label="sync success")
x,y = series_timeout(data["batch-natural-nosa-nore-async"])
axl.plot(x, y, marker="^", label="sync success")

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
