import json, os
import matplotlib.pyplot as plt
import numpy as np

PIPES=4
RANK=0

data = {
  '500' : {
    'ns' : {
      0 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "ns.t-500.rank-0.json"))),
      1 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "ns.t-500.rank-1.json")))
    },
    'su' : {
      0 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "su.straggle-10-10-2000.t-500-1500.rank-0.json"))),
      1 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "su.straggle-10-10-2000.t-500-1500.rank-1.json"))),
    },
    "sa-3000" : {
      0 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "sa.straggle-10-10-2000.t-500-3000.rank-0.json"))),
      1 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "sa.straggle-10-10-2000.t-500-3000.rank-1.json"))),
    },
    "sa-1500" : {
      0 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "sa.straggle-10-10-2000.t-500-1500.rank-0.json"))),
      1 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "sa.straggle-10-10-2000.t-500-1500.rank-1.json"))),
    },
    "sa-1000" : {
      0 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "sa.straggle-10-10-2000.t-500-1000.rank-0.json"))),
      1 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "sa.straggle-10-10-2000.t-500-1000.rank-1.json"))),
    },
    "sa-0750" : {
      0 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "sa.straggle-10-10-2000.t-500-0750.rank-0.json"))),
      1 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "sa.straggle-10-10-2000.t-500-0750.rank-1.json"))),
    },
    "sa-0500" : {
      0 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "sa.straggle-10-10-2000.t-500-0500.rank-0.json"))),
      1 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "sa.straggle-10-10-2000.t-500-0500.rank-1.json"))),
    }
  },
  "1000" : {
    "su" : {
      0 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "su.straggle-10-10-2000.t-1000-1500.rank-0.json"))),
      1 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "su.straggle-10-10-2000.t-1000-1500.rank-1.json"))),
    },
    "sa-1500" : {
      0 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "sa.straggle-10-10-2000.t-1000-1500.rank-0.json"))),
      1 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "sa.straggle-10-10-2000.t-1000-1500.rank-1.json"))),
    }
  },
  "1500" : {
    "sa-1500" : {
      0 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "sa.straggle-10-10-2000.t-1500-1500.rank-0.json"))),
      1 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "sa.straggle-10-10-2000.t-1500-1500.rank-1.json"))),
    }
  },
  "250" : {
    "sa-1500" : {
      0 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "sa.straggle-10-10-2000.t-250-1500.rank-0.json"))),
      1 : json.load(open(os.path.join(os.path.dirname(__file__), f"pipe-{PIPES}", "sa.straggle-10-10-2000.t-250-1500.rank-1.json"))),
    }
  }
}



fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharex=False)
axes = axes.flatten()

xmax = 100

ns = data['500']['ns'][RANK]['data']['times'][:xmax]
su = data['500']['su'][RANK]['data']['times'][:xmax]
sa_3000 = data['500']['sa-3000'][RANK]['data']['times'][:xmax]
sa_1500 = data['500']['sa-1500'][RANK]['data']['times'][:xmax]
sa_1000 = data['500']['sa-1000'][RANK]['data']['times'][:xmax]
sa_0750 = data['500']['sa-0750'][RANK]['data']['times'][:xmax]
sa_0500 = data['500']['sa-0500'][RANK]['data']['times'][:xmax]

axes[0].plot(range(xmax), ns, 'd-', color='tab:green', linewidth=2, markersize=4, label="baseline-ns")
axes[0].plot(range(xmax), su, 'd-', color='tab:red', linewidth=2, markersize=4, label="baseline-su")
axes[0].plot(range(xmax), sa_3000, 's-', color='midnightblue', linewidth=2, markersize=4, label="0.5/3.0 ms")
axes[0].plot(range(xmax), sa_1500, 's-', color='blue', linewidth=2, markersize=4, label="0.5/1.5 ms")
axes[0].plot(range(xmax), sa_1000, 's-', color='tab:blue', linewidth=2, markersize=4, label="0.5/1.0 ms")
axes[0].plot(range(xmax), sa_0750, 's-', color='dodgerblue', linewidth=2, markersize=4, label="0.5/0.75 ms")
axes[0].plot(range(xmax), sa_0500, 's-', color='deepskyblue', linewidth=2, markersize=4, label="0.5/0.5 ms")


# ns_1 = data['500']['baseline-ns'][1]['data']['times'][:xmax]
su           = data['500']['su'][RANK]['data']['times'][:xmax]
sa_1500_1500 = data['1500']['sa-1500'][RANK]['data']['times'][:xmax]
sa_1000_1500 = data['1000']['sa-1500'][RANK]['data']['times'][:xmax]
sa_0500_1500 = data['500']['sa-1500'][RANK]['data']['times'][:xmax]
sa_0250_1500 = data['250']['sa-1500'][RANK]['data']['times'][:xmax]



axes[1].plot(range(xmax), su, 'd-', color='tab:red', linewidth=2, markersize=4, label="baseline-su")
axes[1].plot(range(xmax), sa_1500_1500, 's-', color='midnightblue', linewidth=2, markersize=4, label="1.5/1.5 ms")
axes[1].plot(range(xmax), sa_1000_1500, 's-', color='blue', linewidth=2, markersize=4, label="1.0/1.5 ms")
axes[1].plot(range(xmax), sa_0500_1500, 's-', color='tab:blue', linewidth=2, markersize=4, label="0.5/1.5 ms")
axes[1].plot(range(xmax), sa_0250_1500, 's-', color='dodgerblue', linewidth=2, markersize=4, label="0.25/1.5 ms")



axes[0].set_ylabel('Operation Latency (ms)')
axes[0].set_xlabel("Operation Index")
axes[0].set_title(f"Fixed RTT timeout, varying straggler timeout")

axes[1].set_ylabel('Operation Latency (ms)')
axes[1].set_xlabel("Operation Index")
axes[1].set_title(f"Fixed straggler timeout, varying RTT timeout")

axes[2].set_ylabel('Operation Latency (ms)')
axes[2].set_xlabel('Time')
axes[3].set_ylabel('Operation Latency (ms)')
axes[3].set_xlabel('Time')



# Panel [2]: Fixed RTT timeout, varying straggler timeout (same series as axes[0])
cum_ns      = np.cumsum(ns)
cum_su_500  = np.cumsum(su)           # su defined earlier from data['500']
cum_sa_3000 = np.cumsum(sa_3000)
cum_sa_1500 = np.cumsum(sa_1500)
cum_sa_1000 = np.cumsum(sa_1000)
cum_sa_0750 = np.cumsum(sa_0750)
cum_sa_0500 = np.cumsum(sa_0500)

axes[2].plot(cum_ns,      ns,      'd-', color='tab:green',     linewidth=2, markersize=4, label="baseline-ns")
axes[2].plot(cum_su_500,  su,      'd-', color='tab:red',       linewidth=2, markersize=4, label="baseline-su")
axes[2].plot(cum_sa_3000, sa_3000, 's-', color='midnightblue',  linewidth=2, markersize=4, label="0.5/3.0 ms")
axes[2].plot(cum_sa_1500, sa_1500, 's-', color='blue',          linewidth=2, markersize=4, label="0.5/1.5 ms")
axes[2].plot(cum_sa_1000, sa_1000, 's-', color='tab:blue',      linewidth=2, markersize=4, label="0.5/1.0 ms")
axes[2].plot(cum_sa_0750, sa_0750, 's-', color='dodgerblue',    linewidth=2, markersize=4, label="0.5/0.75 ms")
axes[2].plot(cum_sa_0500, sa_0500, 's-', color='deepskyblue',   linewidth=2, markersize=4, label="0.5/0.5 ms")

# Panel [3]: Fixed straggler timeout, varying RTT timeout (same series as axes[1])
# (su was reassigned above to the 500-series; reuse it here intentionally)
cum_su_for_1500    = np.cumsum(su)
cum_1500_1500      = np.cumsum(sa_1500_1500)
cum_1000_1500      = np.cumsum(sa_1000_1500)
cum_0500_1500      = np.cumsum(sa_0500_1500)
cum_0250_1500      = np.cumsum(sa_0250_1500)

axes[3].plot(cum_su_for_1500, su,             'd-', color='tab:red',      linewidth=2, markersize=4, label="baseline-su")
axes[3].plot(cum_1500_1500,   sa_1500_1500,   's-', color='midnightblue', linewidth=2, markersize=4, label="1.5/1.5 ms")
axes[3].plot(cum_1000_1500,   sa_1000_1500,   's-', color='blue',         linewidth=2, markersize=4, label="1.0/1.5 ms")
axes[3].plot(cum_0500_1500,   sa_0500_1500,   's-', color='tab:blue',     linewidth=2, markersize=4, label="0.5/1.5 ms")
axes[3].plot(cum_0250_1500,   sa_0250_1500,   's-', color='dodgerblue',   linewidth=2, markersize=4, label="0.25/1.5 ms")

# Update bottom x-axis labels to reflect accumulated time
axes[2].set_xlabel('Accumulated Time (ms)')
axes[3].set_xlabel('Accumulated Time (ms)')

for ax in axes:
  # ax.set_xlim(0, xmax)
  ax.grid(True, alpha=0.3)
  ax.legend()

plt.suptitle(f"Allreduce latency with various timeout configurations (rank {RANK})", fontweight='bold')
plt.tight_layout()
plt.show()
