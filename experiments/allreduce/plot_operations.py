import matplotlib.pyplot as plt
from data import OPER
import numpy as np


# pipes|threads|window|tx|rx|type
dat = OPER[4][6][64][64][64]



fig, axes = plt.subplots(1,2, figsize=(15, 8))
plot_style = {
    'linestyle': '-',
    'marker': 'o',       # Use 'o' or '.'
    'alpha': 0.6,        # 0.0 is invisible, 1.0 is solid. 0.6 is a sweet spot.
    'linewidth': 0.8,    # Thinner lines reduce clutter
    'markersize': 2      # Tiny markers so they don't hide the line
}

styles = [
    {'color': 'green', 'linestyle': '-', 'marker': 'o', 'linewidth': 0.8,  'markersize': 8 },
    {'linestyle': '-', 'marker': 'o', 'linewidth': 0.8,  'markersize': 5 },
    {'color': 'red', 'linestyle': '-', 'marker': 's', 'linewidth': 0.5,  'markersize': 5 },

    # {'linestyle': '-', 'marker': 'd', 'linewidth': 0.5,  'markersize': 5 },
    {'linestyle': '-', 'marker': 'x', 'linewidth': 0.5,  'markersize': 6 },
    {'linestyle': '-', 'marker': 'x', 'linewidth': 0.5,  'markersize': 6 },
    {'linestyle': '-', 'marker': 'x', 'linewidth': 0.5,  'markersize': 6 },
    {'linestyle': '-', 'marker': 'x', 'linewidth': 0.5,  'markersize': 6 },
    {'linestyle': '-', 'marker': 'x', 'linewidth': 0.5,  'markersize': 6 },
    {'linestyle': '-', 'marker': 'x', 'linewidth': 0.5,  'markersize': 6 },
    {'linestyle': '-', 'marker': 'x', 'linewidth': 0.5,  'markersize': 6 },
    {'linestyle': '-', 'marker': 'x', 'linewidth': 0.5,  'markersize': 6 },
    {'linestyle': '-', 'marker': 'x', 'linewidth': 0.5,  'markersize': 6 },
    {'linestyle': '-', 'marker': 'X', 'linewidth': 0.5,  'markersize': 6 },
    {'linestyle': '-', 'marker': 'X', 'linewidth': 0.5,  'markersize': 6 },
    {'linestyle': '-', 'marker': 'X', 'linewidth': 0.5,  'markersize': 6 },
    {'linestyle': '-', 'marker': 'X', 'linewidth': 0.5,  'markersize': 6 },
]
# ---------------------------------------------------------
# Left: Float (X = Operation Index, Y = Duration)
# ---------------------------------------------------------
ax = axes[0]
ax.set_title("Float: Duration per Operation")
for i, (label, data) in enumerate(dat.items()):
    print(i)
    times = data['times']
    ax.plot(range(len(times)), times, label=label, **styles[i])
    ax.grid(True, alpha=0.3)

ax.set_ylabel("Duration")
ax.set_xlabel("Operation ID")
ax.legend()

# ---------------------------------------------------------
# Right: Float (X = Cumulative Time, Y = Duration)
# ---------------------------------------------------------
ax = axes[1]
ax.set_title("Float: Duration vs Cumulative Time")
for i, (label, data) in enumerate(dat.items()):
    times = data['times']
    
    # Calculate the cumulative sum of the times for the X-axis
    cumulative_times = np.cumsum(times)
    ax.plot(cumulative_times, times, label=label, **styles[i])
    ax.grid(True, alpha=0.3)

ax.set_ylabel("Duration")
ax.set_xlabel("Cumulative Time")
ax.legend()

plt.tight_layout()
plt.show()


# plt.title("Operation Times")
# plt.xlabel("Operation ID")
# plt.ylabel("Time (ms)")
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# plt.show()