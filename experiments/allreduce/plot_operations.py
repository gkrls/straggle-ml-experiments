import matplotlib.pyplot as plt
from data import OPER
import numpy as np


# pipes|threads|window|tx|rx|type
pipes,wnd,tx,rx = 4,64,64,64
dat = OPER[pipes][6][wnd][tx][rx]

fig, axes = plt.subplots(2,2, figsize=(15, 8))
plot_style = {
    'linestyle': '-',
    'marker': 'o',       # Use 'o' or '.'
    'alpha': 0.6,        # 0.0 is invisible, 1.0 is solid. 0.6 is a sweet spot.
    'linewidth': 0.8,    # Thinner lines reduce clutter
    'markersize': 2      # Tiny markers so they don't hide the line
}

def get_style(label):
  if label == "no.no": 
     return {'color': 'green', 'marker': 'o', 'linewidth': 0.8,  'markersize': 8 }
  if label.startswith("no."): 
     return {'color': 'red', 'marker': 's', 'linewidth': 2.5,  'markersize': 5 }
  if label.startswith("5-125"):
     return {'marker': 'X', 'linewidth': 0.5,  'markersize': 6 }
  if label.startswith("5-250"):
     return {'marker': 'x', 'linewidth': 0.5,  'markersize': 6 }
  if label.startswith("5-500"):
     return {'marker': 'd', 'linewidth': 0.5,  'markersize': 6 }
  return {}

styles = [
    {'color': 'green', 'marker': 'o', 'linewidth': 0.8,  'markersize': 8 },
    {'color': 'blue', 'marker': 'o', 'linewidth': 0.6,  'markersize': 5 },
    {'color': 'red', 'marker': 's', 'linewidth': 2.5,  'markersize': 5 },

    # {'linestyle': '-', 'marker': 'd', 'linewidth': 0.5,  'markersize': 5 },
    {'marker': 'x', 'linewidth': 0.5,  'markersize': 6 },
    {'marker': 'x', 'linewidth': 0.5,  'markersize': 6 },
    {'marker': 'x', 'linewidth': 0.5,  'markersize': 6 },
    {'marker': 'x', 'linewidth': 0.5,  'markersize': 6 },
    {'marker': 'x', 'linewidth': 0.5,  'markersize': 6 },
    {'marker': 'x', 'linewidth': 0.5,  'markersize': 6 },
    {'marker': 'x', 'linewidth': 0.5,  'markersize': 6 },
    {'marker': 'x', 'linewidth': 0.5,  'markersize': 6 },
    {'marker': 'x', 'linewidth': 0.5,  'markersize': 6 },
    {'marker': 'x', 'linewidth': 0.5,  'markersize': 6 },
    {'marker': 'X', 'linewidth': 0.5,  'markersize': 6 },
    {'marker': 'X', 'linewidth': 0.5,  'markersize': 6 },
    {'marker': 'X', 'linewidth': 0.5,  'markersize': 6 },
    {'marker': 'X', 'linewidth': 0.5,  'markersize': 6 },
    {'marker': 'X', 'linewidth': 0.5,  'markersize': 6 },
    {'marker': 'X', 'linewidth': 0.5,  'markersize': 6 },
]
# ---------------------------------------------------------
# Top-Left: Float (X = Operation Index, Y = Duration)
# ---------------------------------------------------------
ax = axes[0,0]
ax.set_title(f"Duration per Operation - dpa{pipes * 64}-{wnd}-{tx}/{rx}")
ax.set_ylim(0,200)
for i, (label, data) in enumerate(dat.items()):
    if not label.startswith('no.') and label.endswith('.no'): continue
    times = data['times']
    style = get_style(label)
    print("style", style)
    label = label[:-4] if label.endswith("yes") else label
    ax.plot(range(len(times)), times, label=label, **style)
    ax.grid(True, alpha=0.3)

ax.set_ylabel("Duration")
ax.set_xlabel("Operation ID")
ax.legend()

# ---------------------------------------------------------
# Bottom-Left: Float (X = Cumulative Time, Y = Duration)
# ---------------------------------------------------------
ax = axes[1,0]
ax.set_title("Float: Duration vs Cumulative Time")
ax.set_ylim(0,400)
for i, (label, data) in enumerate(dat.items()):
    if not label.startswith('no.') and label.endswith('.no'): continue
    times = data['times']
    # Calculate the cumulative sum of the times for the X-axis
    cumulative_times = np.cumsum(times)
    ax.plot(cumulative_times, times, label=label[:-4] if label.endswith("yes") else label, **styles[i])
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