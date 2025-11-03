import json, os
import matplotlib.pyplot as plt



data = {
  '500' : {
    'baseline-ns' : {
      0 : json.load(open(os.path.join(os.path.dirname(__file__), "baseline.t-500.rank-0.json"))),
      1 : json.load(open(os.path.join(os.path.dirname(__file__), "baseline.t-500.rank-1.json")))
    },
    'baseline-su' : {
      0 : json.load(open(os.path.join(os.path.dirname(__file__), "baseline.straggle-10-10-2000.t-500-1500.rank-0.json"))),
      1 : json.load(open(os.path.join(os.path.dirname(__file__), "baseline.straggle-10-10-2000.t-500-1500.rank-1.json"))),
    },
    "sa-3000" : {
      0 : json.load(open(os.path.join(os.path.dirname(__file__), "sa.straggle-10-10-2000.t-500-3000.rank-0.json"))),
      1 : json.load(open(os.path.join(os.path.dirname(__file__), "sa.straggle-10-10-2000.t-500-3000.rank-1.json"))),
    },
    "sa-1500" : {
      0 : json.load(open(os.path.join(os.path.dirname(__file__), "sa.straggle-10-10-2000.t-500-1500.rank-0.json"))),
      1 : json.load(open(os.path.join(os.path.dirname(__file__), "sa.straggle-10-10-2000.t-500-1500.rank-1.json"))),
    },
    "sa-1000" : {
      0 : json.load(open(os.path.join(os.path.dirname(__file__), "sa.straggle-10-10-2000.t-500-1000.rank-0.json"))),
      1 : json.load(open(os.path.join(os.path.dirname(__file__), "sa.straggle-10-10-2000.t-500-1000.rank-1.json"))),
    },
    "sa-0750" : {
      0 : json.load(open(os.path.join(os.path.dirname(__file__), "sa.straggle-10-10-2000.t-500-0750.rank-0.json"))),
      1 : json.load(open(os.path.join(os.path.dirname(__file__), "sa.straggle-10-10-2000.t-500-0750.rank-1.json"))),
    },
    "sa-0500" : {
      0 : json.load(open(os.path.join(os.path.dirname(__file__), "sa.straggle-10-10-2000.t-500-0500.rank-0.json"))),
      1 : json.load(open(os.path.join(os.path.dirname(__file__), "sa.straggle-10-10-2000.t-500-0500.rank-1.json"))),
    }
  }
}



fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=False)
axes = axes.flatten()

xmax = 100

ns_0 = data['500']['baseline-ns'][0]['data']['times'][:xmax]
su_0 = data['500']['baseline-su'][0]['data']['times'][:xmax]
sa_3000_0 = data['500']['sa-3000'][0]['data']['times'][:xmax]
sa_1500_0 = data['500']['sa-1500'][0]['data']['times'][:xmax]
sa_1000_0 = data['500']['sa-1000'][0]['data']['times'][:xmax]
sa_0750_0 = data['500']['sa-0750'][0]['data']['times'][:xmax]
sa_0500_0 = data['500']['sa-0500'][0]['data']['times'][:xmax]

axes[0].plot(range(xmax), ns_0, 'd-', color='tab:green', linewidth=2, markersize=4, label="baseline-ns")
axes[0].plot(range(xmax), su_0, 'd-', color='tab:red', linewidth=2, markersize=4, label="baseline-su")
axes[0].plot(range(xmax), sa_3000_0, 's-', color='midnightblue', linewidth=2, markersize=4, label="0.5/3.0 ms")
axes[0].plot(range(xmax), sa_1500_0, 's-', color='blue', linewidth=2, markersize=4, label="0.5/1.5 ms")
axes[0].plot(range(xmax), sa_1000_0, 's-', color='tab:blue', linewidth=2, markersize=4, label="0.5/1.0 ms")
axes[0].plot(range(xmax), sa_0750_0, 's-', color='dodgerblue', linewidth=2, markersize=4, label="0.5/0.75 ms")
axes[0].plot(range(xmax), sa_0500_0, 's-', color='deepskyblue', linewidth=2, markersize=4, label="0.5/0.5 ms")

# axes[0].plot(list(range(len(times_lt_1))), times_lt_1, 'd-', color='tab:red', linewidth=2, markersize=4, label="straggler")
# axes[2].plot(list(range(len(times_lb_0))), times_lb_0, 's-', color='tab:blue', linewidth=2, markersize=4, label="nostraggler")
# axes[2].plot(list(range(len(times_lb_1))), times_lb_1, 'd-', color='tab:red', linewidth=2, markersize=4, label="straggler")
# # right column

ns_1 = data['500']['baseline-ns'][1]['data']['times'][:xmax]
su_1 = data['500']['baseline-su'][1]['data']['times'][:xmax]
sa_3000_1 = data['500']['sa-3000'][1]['data']['times'][:xmax]
sa_1500_1 = data['500']['sa-1500'][1]['data']['times'][:xmax]
sa_1000_1 = data['500']['sa-1000'][1]['data']['times'][:xmax]
sa_0750_1 = data['500']['sa-0750'][1]['data']['times'][:xmax]
sa_0500_1 = data['500']['sa-0500'][1]['data']['times'][:xmax]


axes[1].plot(range(xmax), ns_1, 'd-', color='tab:green', linewidth=2, markersize=4, label="baseline-ns")
axes[1].plot(range(xmax), su_1, 'd-', color='tab:red', linewidth=2, markersize=4, label="baseline-su")
axes[1].plot(range(xmax), sa_3000_1, 's-', color='midnightblue', linewidth=2, markersize=4, label="0.5/3.0 ms")
axes[1].plot(range(xmax), sa_1500_1, 's-', color='blue', linewidth=2, markersize=4, label="0.5/1.5 ms")
axes[1].plot(range(xmax), sa_1000_1, 's-', color='tab:blue', linewidth=2, markersize=4, label="0.5/1.0 ms")
axes[1].plot(range(xmax), sa_0750_1, 's-', color='dodgerblue', linewidth=2, markersize=4, label="0.5/0.75 ms")
axes[1].plot(range(xmax), sa_0500_1, 's-', color='deepskyblue', linewidth=2, markersize=4, label="0.5/0.5 ms")


# axes[1].plot(list(range(len(times_rt_1))), times_rt_1, 'd-', color='tab:red', linewidth=2, markersize=4, label="straggler")
# axes[3].plot(list(range(len(times_rb_0))), times_rb_0, 's-', color='tab:blue', linewidth=2, markersize=4, label="nostraggler")
# axes[3].plot(list(range(len(times_rb_1))), times_rb_1, 'd-', color='tab:red', linewidth=2, markersize=4, label="straggler")



axes[0].set_ylabel('Operation Latency (ms)')
axes[0].set_xlabel("Operation Index")
axes[1].set_xlabel("Operation Index")

axes[2].set_ylabel('Operation Latency (ms)')
axes[0].set_title(f"Rank 0 (Non-straggler)")
axes[1].set_title(f"Rank 1 (Straggler)")
# axes[1].set_title(f"Straggler mitigation ON ({data_rb_0['straggle_num']}x{data_rb_0['straggle_ms']} at rank {data_rb_0['straggle_rank']})")
# axes[2].set_title("Straggler mitigation OFF")
# axes[3].set_title("Straggler mitigation OFF")


for ax in axes:
  # ax.set_xlim(0, xmax)
  ax.grid(True, alpha=0.3)
  ax.legend()
plt.tight_layout()
plt.show()
