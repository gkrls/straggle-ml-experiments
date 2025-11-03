import json, os
import matplotlib.pyplot as plt


data_lt_0 = json.load(open(os.path.join(os.path.dirname(__file__), "data/allreduce-benchmark2.k-5.0-2-5000.rank0.json")))
data_lt_1 = json.load(open(os.path.join(os.path.dirname(__file__), "data/allreduce-benchmark2.k-5.0-2-5000.rank1.json")))
data_lb_0 = json.load(open(os.path.join(os.path.dirname(__file__), "data/allreduce-benchmark2.k-6.0-2-5000.rank0.json")))
data_lb_1 = json.load(open(os.path.join(os.path.dirname(__file__), "data/allreduce-benchmark2.k-6.0-2-5000.rank1.json")))
data_rt_0 = json.load(open(os.path.join(os.path.dirname(__file__), "data/allreduce-benchmark2.k-5.0-4-5000.rank0.json")))
data_rt_1 = json.load(open(os.path.join(os.path.dirname(__file__), "data/allreduce-benchmark2.k-5.0-4-5000.rank1.json")))
data_rb_0 = json.load(open(os.path.join(os.path.dirname(__file__), "data/allreduce-benchmark2.k-5.4-4-5000.rank0.json")))
data_rb_1 = json.load(open(os.path.join(os.path.dirname(__file__), "data/allreduce-benchmark2.k-5.4-4-5000.rank1.json")))
times_lt_0 = data_lt_0['data']['times']
times_lt_1 = data_lt_1['data']['times']
times_lb_0 = data_lb_0['data']['times']
times_lb_1 = data_lb_1['data']['times']
times_rt_0 = data_rt_0['data']['times']
times_rt_1 = data_rt_1['data']['times']
times_rb_0 = data_rb_0['data']['times']
times_rb_1 = data_rb_1['data']['times']

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
axes = axes.flatten()

# left column
axes[0].plot(list(range(len(times_lt_0))), times_lt_0, 's-', color='tab:blue', linewidth=2, markersize=4, label="nostraggler")
axes[0].plot(list(range(len(times_lt_1))), times_lt_1, 'd-', color='tab:red', linewidth=2, markersize=4, label="straggler")
axes[2].plot(list(range(len(times_lb_0))), times_lb_0, 's-', color='tab:blue', linewidth=2, markersize=4, label="nostraggler")
axes[2].plot(list(range(len(times_lb_1))), times_lb_1, 'd-', color='tab:red', linewidth=2, markersize=4, label="straggler")
# right column
axes[1].plot(list(range(len(times_rt_0))), times_rt_0, 's-', color='tab:blue', linewidth=2, markersize=4, label="nostraggler")
axes[1].plot(list(range(len(times_rt_1))), times_rt_1, 'd-', color='tab:red', linewidth=2, markersize=4, label="straggler")
axes[3].plot(list(range(len(times_rb_0))), times_rb_0, 's-', color='tab:blue', linewidth=2, markersize=4, label="nostraggler")
axes[3].plot(list(range(len(times_rb_1))), times_rb_1, 'd-', color='tab:red', linewidth=2, markersize=4, label="straggler")



axes[0].set_ylabel('Time (s)')
axes[2].set_ylabel('Time (s)')
axes[0].set_title(f"Straggler mitigation ON ({data_lt_0['straggle_num']}x{data_lt_0['straggle_ms']} at rank {data_lt_0['straggle_rank']})")
axes[1].set_title(f"Straggler mitigation ON ({data_rb_0['straggle_num']}x{data_rb_0['straggle_ms']} at rank {data_rb_0['straggle_rank']})")
axes[2].set_title("Straggler mitigation OFF")
axes[3].set_title("Straggler mitigation OFF")


for ax in axes:
  ax.grid(True, alpha=0.3)
  ax.legend()
plt.tight_layout()
plt.show()
