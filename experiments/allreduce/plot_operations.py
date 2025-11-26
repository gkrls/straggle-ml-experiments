import matplotlib.pyplot as plt
from data import OPER
import numpy as np
import re


# pipes|threads|window|tx|rx|type
pipes,wnd,tx,rx = 4,64,64,64
dat = OPER[pipes][6][wnd][tx][rx]

fig, axes = plt.subplots(2,3, figsize=(15, 8))
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

# ---------------------------------------------------------
# Top-Left: (X = Operation Index, Y = Duration)
# ---------------------------------------------------------
ax = axes[0,0]
ax.set_title(f"Duration per Operation - dpa{pipes * 64}-{wnd}-{tx}/{rx}")
ax.set_ylim(0,200)
for i, (label, data) in enumerate(dat.items()):
    if not label.startswith('no.') and label.endswith('.no'): continue
    style = get_style(label)
    label = label[:-4] if label.endswith("yes") else label
    ax.plot(range(len(data['times'])), data['times'], label=label, **style)
    ax.grid(True, alpha=0.3)

ax.set_ylabel("Duration")
ax.set_xlabel("Operation ID")
ax.legend()


# ---------------------------------------------------------
# Top-Middle: Heatmap
# ---------------------------------------------------------
from matplotlib.colors import ListedColormap, BoundaryNorm
ax = axes[0,1]
# Regex to capture X and Y from the label (e.g., "5-250/01000us.yes")
problem_configs = [(50,25000)]

heatmap_data = {}
x_values = set()
y_values = set()
pattern = re.compile(r"^\d+-(\d+)/(\d+)us\.yes$")

for label, data in dat.items():
    if (match := pattern.match(label)):
        x, y = int(match.group(1)), int(match.group(2))
        heatmap_data[(x, y)] = np.mean(data['times'])
        x_values.add(x)
        y_values.add(y)

sorted_x = sorted(list(x_values))
sorted_y = sorted(list(y_values))
x_map = {val: i for i, val in enumerate(sorted_x)}
y_map = {val: i for i, val in enumerate(sorted_y)}

# Build the Heatmap Matrix AND the Problem Mask Matrix
heatmap_matrix = np.full((len(sorted_y), len(sorted_x)), np.nan) 
problem_mask = np.full((len(sorted_y), len(sorted_x)), False, dtype=bool) 

for (x, y), mean_time in heatmap_data.items():
    row_idx, col_idx = y_map[y], x_map[x]
    heatmap_matrix[row_idx, col_idx] = mean_time
    if (x, y) in problem_configs:
        problem_mask[row_idx, col_idx] = True

# --- 2. Heatmap Plotting with Red Highlight ---


# Plot the main heatmap (all non-problem data)
im = ax.imshow(
    np.ma.masked_where(problem_mask, heatmap_matrix), # Mask out problem data
    cmap="viridis", 
    aspect='auto', 
    interpolation='nearest'
)

# Plot the problematic points on top in red
problem_values = np.ma.masked_where(~problem_mask, heatmap_matrix) # Only show problem data
red_cmap = ListedColormap(['red']) 
ax.imshow(problem_values, cmap=red_cmap, aspect='auto', interpolation='nearest')


# Configure ticks and labels
ax.set_xticks(np.arange(len(sorted_x)))
ax.set_yticks(np.arange(len(sorted_y)))
ax.set_xticklabels(sorted_x, rotation=45, ha="right")
ax.set_yticklabels(sorted_y)

# Add colorbar and titles
ax.figure.colorbar(im, ax=ax, label="Mean Duration")
ax.set_xlabel("X value (e.g., burst limit)")
ax.set_ylabel("Y value (e.g., delay parameter)")
ax.set_title("Heatmap of Mean Duration (Red = Problem Config)")
# ---------------------------------------------------------
# Bottom-Left: Float (X = Cumulative Time, Y = Duration)
# ---------------------------------------------------------
ax = axes[1,0]
ax.set_title("Float: Duration vs Cumulative Time")
ax.set_ylim(0,400)
for i, (label, data) in enumerate(dat.items()):
    if not label.startswith('no.') and label.endswith('.no'): continue
    style = get_style(label)
    label = label[:-4] if label.endswith("yes") else label
    times = data['times']
    cumulative_times = np.cumsum(times)
    ax.plot(cumulative_times, times, label=label, **style)
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