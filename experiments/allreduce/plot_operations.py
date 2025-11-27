import matplotlib.pyplot as plt
from data import OPER
import numpy as np
import re
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

# pipes|threads|window|tx|rx|type|problems


fig, axes = plt.subplots(3,3, figsize=(35, 16))

def get_style(label):
  if label == "no.no": 
     return {'color': 'green', 'marker': 'o', 'linewidth': 0.8,  'markersize': 8 }
  if label.startswith("no."): 
     return {'color': 'red', 'marker': 'o', 'linewidth': 2.5,  'markersize': 5 }
  if label.startswith("5-100"):
     return {'marker': '|', 'linewidth': 0.4,  'markersize': 3 }
  if label.startswith("5-125"):
     return {'marker': 's', 'linewidth': 0.4,  'markersize': 3 }
  if label.startswith("5-250"):
     return {'marker': 'x', 'linewidth': 0.4,  'markersize': 3 }
  if label.startswith("5-500"):
     return {'marker': 'd', 'linewidth': 0.4,  'markersize': 3 }
  return {}

# def do_plots(axes, row, pipes, threads, wnd, tx, rx):
#   dat = OPER[pipes][threads][wnd][tx][rx]
#   # operations plot
#   ax = axes[row,0]
#   ax.set_title(f"Duration per Operation - dpa{pipes * 64}-{wnd * threads}({threads})-{tx}/{rx}")
#   ax.set_ylim(0,125)
#   for i, (label, data) in enumerate(dat.items()):
#       if not label.startswith('no.') and label.endswith('.no'): continue
#       style = get_style(label)
#       label = label[:-4] if label.endswith("yes") else label
#       ax.plot(range(len(data['times'])), data['times'], label=label, **style)
#       ax.grid(True, alpha=0.3)
#   ax.set_ylabel("Duration")
#   ax.set_xlabel("Operation ID")

#   # operations cumulative
#   ax = axes[row,1]
#   ax.set_title("Duration vs Cumulative Time")
#   ax.set_ylim(0,125)
#   for i, (label, data) in enumerate(dat.items()):
#       if not label.startswith('no.') and label.endswith('.no'): continue
#       style = get_style(label)
#       label = label[:-4] if label.endswith("yes") else label
#       times = data['times']
#       cumulative_times = np.cumsum(times)
#       ax.plot(cumulative_times, times, label=label, **style)
#       ax.grid(True, alpha=0.3)
#   ax.set_ylabel("Duration")
#   ax.set_xlabel("Cumulative Time")


def do_plots(axes, row, pipes, threads, wnd, tx, rx,
             problem_configs,
             include_problem_configs_in_plots=True):
  dat = OPER[pipes][threads][wnd][tx][rx]

  # operations plot
  ax = axes[row,0]
  ax.set_title(f"Duration per Operation - dpa{pipes * 64}-{wnd * threads}({threads})-{tx}/{rx}")
  ax.set_ylim(0,125)
  for i, (label, data) in enumerate(dat.items()):
      if not label.startswith('no.') and label.endswith('.no'):
          continue

      is_problem = label in problem_configs
      if not include_problem_configs_in_plots and is_problem:
          continue

      style = get_style(label)
      display_label = label[:-4] if label.endswith("yes") else label
      ax.plot(range(len(data['times'])), data['times'], label=display_label, **style)
      ax.grid(True, alpha=0.3)
  ax.set_ylabel("Duration")
  ax.set_xlabel("Operation ID")

  # operations cumulative
  ax = axes[row,1]
  ax.set_title("Duration vs Cumulative Time")
  ax.set_ylim(0,125)
  for i, (label, data) in enumerate(dat.items()):
      if not label.startswith('no.') and label.endswith('.no'):
          continue

      is_problem = label in problem_configs
      if not include_problem_configs_in_plots and is_problem:
          continue

      style = get_style(label)
      display_label = label[:-4] if label.endswith("yes") else label
      times = data['times']
      cumulative_times = np.cumsum(times)
      ax.plot(cumulative_times, times, label=display_label, **style)
      ax.grid(True, alpha=0.3)
  ax.set_ylabel("Duration")
  ax.set_xlabel("Cumulative Time")

def do_heatmap(axes, row, pipes, threads, wnd, tx, rx, problem_configs):
  dat = OPER[pipes][threads][wnd][tx][rx]
  ax = axes[row,2]

  pattern = re.compile(r"^\d+-(\d+)/(\d+)us\.yes$")
  coords = {}
  x_vals, y_vals = set(), set()

  # Collect all data points and axis values (from actual data)
  for label, data in dat.items():
      if (match := pattern.match(label)):
          x, y = int(match.group(1)), int(match.group(2))
          coords[(x, y)] = np.mean(data['times'])
          x_vals.add(x); y_vals.add(y)

  # Ensure ALL problem X and Y values are included in the axes,
  # even if there is no data in `dat` for them.
  for plabel in problem_configs:
      pm = pattern.match(plabel)
      if pm:
          x = int(pm.group(1))
          y = int(pm.group(2))
          x_vals.add(x)
          y_vals.add(y)

  # Axis setup
  sorted_x, sorted_y = sorted(list(x_vals)), sorted(list(y_vals))
  x_map = {val: i for i, val in enumerate(sorted_x)}
  y_map = {val: i for i, val in enumerate(sorted_y)}

  # Build the Data Matrix (NaN where no data exists)
  matrix = np.full((len(sorted_y), len(sorted_x)), np.nan)
  for (x, y), val in coords.items():
      matrix[y_map[y], x_map[x]] = val

  # Main heatmap layer
  im = ax.imshow(
      matrix,
      cmap="viridis",
      aspect='auto',
      interpolation='nearest',
      vmin=np.nanmin(matrix),
      vmax=np.nanmax(matrix)
  )

  # Overlay: diagonal-hatched rectangles for all problem configs
  for plabel in problem_configs:
      pm = pattern.match(plabel)
      if not pm:
          continue  # skip labels that don't encode timeouts
      x = int(pm.group(1))
      y = int(pm.group(2))

      i = y_map[y]  # row index
      j = x_map[x]  # column index
      cell_val = matrix[i, j]

      # If there is data: keep heatmap color, draw hatched border.
      # If there is NO data: fill red + hatch.
      if not np.isnan(cell_val):
          facecolor = 'none'
          edgecolor = 'black'
          alpha = 1.0
      else:
          facecolor = 'red'
          edgecolor = 'black'
          alpha = 0.8

      rect = patches.Rectangle(
          (j - 0.5, i - 0.5),  # (x, y) in image coordinates
          1, 1,
          linewidth=1.5,
          edgecolor=edgecolor,
          facecolor=facecolor,
          hatch='///',
          alpha=alpha
      )
      ax.add_patch(rect)

  # Axes and labels
  ax.set_xticks(np.arange(len(sorted_x)))
  ax.set_yticks(np.arange(len(sorted_y)))

  ax.set_xticklabels([str(x) for x in sorted_x], rotation=45, ha="right")
  ax.set_yticklabels([str(y) for y in sorted_y])

  ax.set_xlabel("Worker Timeout (us)")
  ax.set_ylabel("Switch Timeout (us)")
  ax.set_title("Mean Duration Heatmap (Diagonal = Problem Config)")

  # Colorbar based on the main data layer
  cbar = ax.figure.colorbar(im, ax=ax)
  cbar.ax.set_ylabel("Mean Duration", rotation=-90, va="bottom")


# def do_heatmap(axes, row, pipes, threads, wnd, tx, rx, problem_configs):
#   dat = OPER[pipes][threads][wnd][tx][rx]
#   # heatmap
#   ax = axes[row,2]
#   pattern = re.compile(r"^\d+-(\d+)/(\d+)us\.yes$")
#   coords = {}
#   x_vals, y_vals = set(), set()

#   # Collect all data points and axis values
#   for label, data in dat.items():
#       if (match := pattern.match(label)):
#           x, y = int(match.group(1)), int(match.group(2))
#           coords[(x, y)] = np.mean(data['times'])
#           x_vals.add(x); y_vals.add(y)

#   for x, y in problem_configs:
#       x_vals.add(x); y_vals.add(y)

#   # Axis setup
#   sorted_x, sorted_y = sorted(list(x_vals)), sorted(list(y_vals))
#   x_map = {val: i for i, val in enumerate(sorted_x)}
#   y_map = {val: i for i, val in enumerate(sorted_y)}

#   # Build the Main Data Matrix (NaN where no data exists)
#   matrix = np.full((len(sorted_y), len(sorted_x)), np.nan) 
#   for (x, y), val in coords.items():
#       matrix[y_map[y], x_map[x]] = val

#   # Build the Problem Highlight Matrix (True where red is needed)
#   problem_matrix = np.full((len(sorted_y), len(sorted_x)), False, dtype=bool)
#   for x, y in problem_configs:
#       problem_matrix[y_map[y], x_map[x]] = True

#   im = ax.imshow(
#       np.ma.masked_where(problem_matrix, matrix), 
#       cmap="viridis", 
#       aspect='auto', 
#       interpolation='nearest',
#       # Set vmin/vmax to ensure consistent color scale across all plots
#       vmin=np.nanmin(matrix), 
#       vmax=np.nanmax(matrix)
#   )

#   # Layer 2: Plot the problem areas (masked to only show where problems exist)
#   # Use a custom colormap that is ONLY red. We use a dummy value (1) for plotting.
#   problem_highlight = np.ma.masked_where(~problem_matrix, np.ones_like(matrix))
#   red_cmap = ListedColormap(['red']) 
#   ax.imshow(problem_highlight, cmap=red_cmap, aspect='auto', interpolation='nearest')


#   # --- 3. Configure Axes and Labels ---
#   ax.set_xticks(np.arange(len(sorted_x)))
#   ax.set_yticks(np.arange(len(sorted_y)))

#   ax.set_xticklabels([str(x) for x in sorted_x], rotation=45, ha="right")
#   ax.set_yticklabels([str(y) for y in sorted_y])

#   ax.set_xlabel("Worker Timeout (us)")
#   ax.set_ylabel("Switch Timeout (us)")
#   ax.set_title("Mean Duration Heatmap (Red = Problem Config)")

#   # Add colorbar (only based on the main data layer)
#   cbar = ax.figure.colorbar(im, ax=ax)
#   cbar.ax.set_ylabel("Mean Duration", rotation=-90, va="bottom")


# def do_row(axes, row, pipes, threads, wnd, tx, rx, problem_configs):
#   do_plots(axes, row, pipes, threads, wnd, tx, rx)
#   do_heatmap(axes, row, pipes, threads, wnd, tx, rx, problem_configs)

def do_row(axes, row, pipes, threads, wnd, tx, rx, problem_configs,
           include_problem_configs_in_plots=True):
  do_plots(axes, row, pipes, threads, wnd, tx, rx,
           problem_configs,
           include_problem_configs_in_plots=include_problem_configs_in_plots)
  do_heatmap(axes, row, pipes, threads, wnd, tx, rx, problem_configs)


problem_configs_row_0 = [
  "5-050/00500us.yes",
  "5-050/00750us.yes",
  "5-075/00500us.yes",
  "5-075/00750us.yes",
  "5-100/00500us.yes"
  # "5-125/00500us.yes", # sometimes
]

row_0 = 4,6,64,64,64,problem_configs_row_0
do_row(axes, 0, *row_0, False)


problem_configs_row_1 =[
  "5-050/00125us.yes",
  "5-050/00250us.yes",
  "5-050/00500us.yes",
  "5-050/00750us.yes",
]
row_1= 4,6,32,32,32,problem_configs_row_1
do_row(axes, 1, *row_1, False)


do_row(axes, 2, *row_0, False)

fig.subplots_adjust(
    left=0.1,   # space for legend
    right=0.98,  # reduce right whitespace
    top=0.98,    # reduce top whitespace
    bottom=0.02,  # reduce bottom whitespace
    wspace=0.15,   # horizontal space BETWEEN columns
    hspace=0.25    # vertical space BETWEEN rows
)

# 2) Get handles/labels from the first subplot
handles, labels = axes[0,0].get_legend_handles_labels()

# In case matplotlib has hidden some internal labels, filter out the "_nolegend_" stuff
handles_filtered = []
labels_filtered = []
for h, lab in zip(handles, labels):
    if not lab.startswith('_'):
        handles_filtered.append(h)
        labels_filtered.append(lab)

# 3) Create ONE legend in figure coordinates, on the left, vertically centered
fig.legend(
    handles_filtered,
    labels_filtered,
    loc="center left",
    bbox_to_anchor=(0.00, 0.5),     # (x, y) in figure coords
    frameon=False,
)
# ---------------------------------------------------------
# Top-Left: (X = Operation Index, Y = Duration)
# ---------------------------------------------------------




# ---------------------------------------------------------
# Middle-Left: Heatmap
# ---------------------------------------------------------


# ---------------------------------------------------------------------

# --- 1. Data Preparation ---



# ---------------------------------------------------------
# Bottom-Left: Float (X = Cumulative Time, Y = Duration)
# ---------------------------------------------------------




# plt.tight_layout()
plt.show()




# plt.title("Operation Times")
# plt.xlabel("Operation ID")
# plt.ylabel("Time (ms)")
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# plt.show()