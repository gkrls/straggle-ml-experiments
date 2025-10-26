# import json, os
# import matplotlib.pyplot as plt
# import numpy as np

# FILE1='../experiments/train/results/gpt2_straggle_16.json'
# FILE2='../experiments/train/results/gpt2.json'
# FILE1=os.path.join(os.path.dirname(__file__), FILE1)
# FILE2=os.path.join(os.path.dirname(__file__), FILE2)

# METRIC="val_ppl"
# MINIVAL_METRIC="mini_val_ppl" 
# YLABEL="Perplexity"  
# TARGET=30.0  # Target perplexity
# LOWER_IS_BETTER=True  # True for perplexity, loss; False for accuracy
# USE_MINIVAL=True  # Set to False to only use epoch data

import json, os
import matplotlib.pyplot as plt
import numpy as np

FILE1='../experiments/train/results/gpt2_straggle_16.json'
FILE2='../experiments/train/results/gpt2.json'
FILE1=os.path.join(os.path.dirname(__file__), FILE1)
FILE2=os.path.join(os.path.dirname(__file__), FILE2)

# Configure your metric here
METRIC="val_ppl"  # Change to match your data: val_ppl, val_loss, etc.
MINIVAL_METRIC="mini_val_ppl"  # The corresponding minival metric
YLABEL="Perplexity"  
TARGET=30.0  # Target perplexity
LOWER_IS_BETTER=True  # True for perplexity, loss; False for accuracy
USE_MINIVAL=True  # Set to False to only use epoch data

# Read the first dataset (baseline)
with open(FILE1, 'r') as f:
    data1 = json.loads(f.read())

# Read the second dataset
with open(FILE2, 'r') as f:
    data2 = json.loads(f.read())

def extract_combined_data(epochs_data, minival_data=None, use_minival=True):
    """Extract both epoch and minival data points with interpolated times"""
    all_points = []  # List of (global_step, metric, cumulative_time) tuples
    
    # First pass: get epoch data for time calculations
    epoch_times = {}
    cumulative_time = 0.0
    for epoch_key in sorted(epochs_data.keys(), key=int):
        epoch_data = epochs_data[epoch_key]
        steps = epoch_data['steps']
        step_time = epoch_data['step_time']
        epoch_time = steps * step_time
        epoch_times[int(epoch_key)] = {
            'start_time': cumulative_time,
            'end_time': cumulative_time + epoch_time,
            'step_time': step_time,
            'total_steps': steps
        }
        cumulative_time += epoch_time
    
    # Add minival points if requested and available
    if use_minival and minival_data:
        for epoch_key in sorted(minival_data.keys(), key=int):
            epoch_num = int(epoch_key)
            if epoch_num not in epoch_times:
                continue
                
            epoch_info = epoch_times[epoch_num]
            epoch_minival = minival_data[epoch_key]
            
            # Add each minival checkpoint
            for step_key in sorted(epoch_minival.keys()):
                step_data = epoch_minival[step_key]
                global_step = step_data['global_step']
                metric_value = step_data[MINIVAL_METRIC]
                
                # Interpolate time based on step position within epoch
                step_within_epoch = int(step_key)  # 300, 600, 900, or 1200
                time_fraction = step_within_epoch / epoch_info['total_steps']
                interpolated_time = (epoch_info['start_time'] + 
                                   time_fraction * (epoch_info['end_time'] - epoch_info['start_time']))
                
                all_points.append((global_step, metric_value, interpolated_time))
    
    # Add epoch-end points (these might duplicate the last minival point)
    for epoch_key in sorted(epochs_data.keys(), key=int):
        epoch_data = epochs_data[epoch_key]
        epoch_num = int(epoch_key)
        global_step = epoch_data['global_step']
        metric_value = epoch_data[METRIC]
        time_value = epoch_times[epoch_num]['end_time']
        
        # Only add if not a duplicate of last minival point
        if not all_points or all_points[-1][0] != global_step:
            all_points.append((global_step, metric_value, time_value))
    
    # Sort by global_step and extract arrays
    all_points.sort(key=lambda x: x[0])
    global_steps = [p[0] for p in all_points]
    metrics = [p[1] for p in all_points]
    times = [p[2] for p in all_points]
    
    return global_steps, metrics, times

def extract_epoch_only_data(epochs_data):
    """Extract only epoch-level data (fallback if no minival)"""
    epoch_numbers = []
    epoch_metrics = []
    cumulative_times = []
    
    cumulative_time = 0.0
    for epoch_key in sorted(epochs_data.keys(), key=int):
        epoch_data = epochs_data[epoch_key]
        
        # Calculate time for this epoch
        steps = epoch_data['steps']
        step_time = epoch_data['step_time']
        epoch_time = steps * step_time
        cumulative_time += epoch_time
        
        # Store data
        epoch_numbers.append(int(epoch_key))
        epoch_metrics.append(epoch_data[METRIC])
        cumulative_times.append(cumulative_time)
    
    return epoch_numbers, epoch_metrics, cumulative_times

# Extract data for both datasets
minival1 = data1.get('minival', None) if USE_MINIVAL else None
minival2 = data2.get('minival', None) if USE_MINIVAL else None

if minival1 or minival2:
    steps1, metrics1, times1 = extract_combined_data(data1['epochs'], minival1, USE_MINIVAL)
    steps2, metrics2, times2 = extract_combined_data(data2['epochs'], minival2, USE_MINIVAL)
else:
    # Fallback to epoch-only data
    steps1, metrics1, times1 = extract_epoch_only_data(data1['epochs'])
    steps2, metrics2, times2 = extract_epoch_only_data(data2['epochs'])

# Convert to minutes
times1_min = np.array(times1) / 60
times2_min = np.array(times2) / 60

# Auto-detect metric direction if not explicitly set
if METRIC in ['perplexity', 'ppl', 'val_ppl', 'mini_val_ppl', 'loss', 'val_loss', 'mini_val_loss']:
    LOWER_IS_BETTER = True
elif METRIC in ['accuracy', 'acc', 'val_acc', 'top1', 'top5', 'val_top1', 'val_top5']:
    LOWER_IS_BETTER = False

# Find when each hits target
def find_metric_target(metrics, metric_target, times_min, steps, lower_is_better=False):
    """Find when metric crosses target threshold"""
    for i, m in enumerate(metrics):
        if lower_is_better:
            if m <= metric_target:
                return times_min[i], steps[i], i
        else:
            if m >= metric_target:
                return times_min[i], steps[i], i
    return None, None, None

time1_target, step1_target, idx1_target = find_metric_target(
    metrics1, TARGET, times1_min, steps1, LOWER_IS_BETTER)
time2_target, step2_target, idx2_target = find_metric_target(
    metrics2, TARGET, times2_min, steps2, LOWER_IS_BETTER)

# Truncate data to show only a few points after hitting target
def truncate_after_target(times, metrics, idx_target, extra_points=5):
    if idx_target is not None:
        end_idx = min(idx_target + extra_points, len(times))
        return times[:end_idx], metrics[:end_idx]
    return times, metrics

times1_plot, metrics1_plot = truncate_after_target(times1_min, metrics1, idx1_target, 8)
times2_plot, metrics2_plot = truncate_after_target(times2_min, metrics2, idx2_target, 8)

# Calculate speedup
speedup_text = ""
mid_x = 0
mid_y = 0
y_range = 0
if time1_target and time2_target:
    speedup = time1_target / time2_target
    time_saved = time1_target - time2_target
    speedup_text = f'{speedup:.2f}x faster'

# Create the plot
plt.figure(figsize=(12, 7))

# Plot both lines with smaller markers for intermediate points
marker_size = 4 if USE_MINIVAL else 5
plt.plot(times1_plot, metrics1_plot, marker='o', linewidth=2.5, markersize=marker_size, 
         color='#D62828', label='Baseline', alpha=0.9)
plt.plot(times2_plot, metrics2_plot, marker='s', linewidth=2.5, markersize=marker_size, 
         color='#1E88E5', label='Optimized', alpha=0.9)

# Add horizontal line at target
target_label = f'Target {"≤" if LOWER_IS_BETTER else "≥"} {TARGET}'
plt.axhline(y=TARGET, color='gray', linestyle='--', linewidth=1.5, 
            label=target_label, alpha=0.7)

# Add vertical lines when each hits target
y_bottom = plt.ylim()[0]
if time1_target:
    plt.axvline(x=time1_target, color='#D62828', linestyle=':', linewidth=1.5, alpha=0.6)
    plt.text(time1_target, y_bottom + (TARGET - y_bottom) * 0.05, 
             f'{time1_target:.1f} min', 
             rotation=90, verticalalignment='bottom', horizontalalignment='right',
             color='#D62828', fontsize=9, fontweight='bold')

if time2_target:
    plt.axvline(x=time2_target, color='#1E88E5', linestyle=':', linewidth=1.5, alpha=0.6)
    plt.text(time2_target, y_bottom + (TARGET - y_bottom) * 0.05, 
             f'{time2_target:.1f} min', 
             rotation=90, verticalalignment='bottom', horizontalalignment='right',
             color='#1E88E5', fontsize=9, fontweight='bold')

# Add speedup annotation
if time1_target and time2_target:
    mid_x = (time1_target + time2_target) / 2
    
    # Position at middle of plot
    y_range = max(max(metrics1_plot), max(metrics2_plot)) - min(min(metrics1_plot), min(metrics2_plot))
    y_min_plot = min(min(metrics1_plot), min(metrics2_plot))
    mid_y = y_min_plot + y_range * 0.5
    
    plt.annotate('', xy=(time2_target, mid_y), xytext=(time1_target, mid_y),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    plt.text(mid_x, mid_y + y_range * 0.08, speedup_text, 
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFA500', edgecolor='black', 
                      linewidth=1.5, alpha=0.85))

# Formatting
plt.xlabel('Time (minutes)', fontsize=12, fontweight='bold')
plt.ylabel(f'{YLABEL}', fontsize=12, fontweight='bold')

# Title includes whether we're using minival data
data_type = "with Intermediate Checkpoints" if USE_MINIVAL and (minival1 or minival2) else ""
direction_text = "Lower" if LOWER_IS_BETTER else "Higher"
plt.title(f'Training Progress Comparison: Baseline vs Optimized\n{data_type} ({direction_text} is Better)', 
          fontsize=14, fontweight='bold', pad=20)

plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=10, loc='best')

# Set y-axis limits based on data
if LOWER_IS_BETTER:
    # For decreasing metrics, show from max down to slightly below minimum
    y_min = min(min(metrics1_plot), min(metrics2_plot), TARGET) * 0.95
    y_max = max(metrics1_plot[0], metrics2_plot[0]) * 1.05
    plt.ylim(y_min, y_max)
else:
    # For increasing metrics
    y_min = max(0, min(min(metrics1_plot), min(metrics2_plot)) * 0.95)
    y_max = min(100, max(max(metrics1_plot), max(metrics2_plot)) * 1.05)
    plt.ylim(y_min, y_max)

plt.tight_layout()

# Save the plot (uncomment if needed)
# plt.savefig('/mnt/user-data/outputs/time_to_metric_comparison.png', dpi=300, bbox_inches='tight')
# print(f"Plot saved to /mnt/user-data/outputs/time_to_metric_comparison.png")

# Print statistics
print(f"\n{'='*60}")
print(f"TIME-TO-METRIC COMPARISON")
print(f"{'='*60}")

if time1_target:
    print(f"\nBaseline:")
    print(f"  - Reached {TARGET} at step {step1_target}: {time1_target:.2f} minutes")
    print(f"  - Final {YLABEL}: {metrics1[-1]:.2f}")
    print(f"  - Total time: {times1_min[-1]:.2f} minutes")

if time2_target:
    print(f"\nOptimized:")
    print(f"  - Reached {TARGET} at step {step2_target}: {time2_target:.2f} minutes")
    print(f"  - Final {YLABEL}: {metrics2[-1]:.2f}")
    print(f"  - Total time: {times2_min[-1]:.2f} minutes")

if time1_target and time2_target:
    print(f"\n{'='*60}")
    print(f"SPEEDUP RESULTS")
    print(f"{'='*60}")
    print(f"  - Time saved to {TARGET}: {time_saved:.2f} minutes ({time_saved/60:.2f} hours)")
    print(f"  - Speedup factor: {speedup:.2f}x")
    print(f"  - Percentage improvement: {(1 - 1/speedup)*100:.1f}%")
    print(f"{'='*60}")

# Print coordinates for LaTeX
print(f"\n{'='*60}")
print(f"PLOT COORDINATES (for LaTeX)")
print(f"{'='*60}")
print(f"Metric: {METRIC} ({'lower' if LOWER_IS_BETTER else 'higher'} is better)")
print(f"Target: {TARGET}")
print(f"Data points: {'With minival checkpoints' if USE_MINIVAL and (minival1 or minival2) else 'Epoch-level only'}")

print(f"\nBaseline coordinates:")
print(f"% Time (min), {YLABEL}")
for t, m in zip(times1_plot, metrics1_plot):
    print(f"({t:.2f}, {m:.2f})")

print(f"\nOptimized coordinates:")
print(f"% Time (min), {YLABEL}")
for t, m in zip(times2_plot, metrics2_plot):
    print(f"({t:.2f}, {m:.2f})")

print(f"\nTarget line:")
print(f"% Horizontal line at y={TARGET}")
# Fix for numpy array truth value issue
max_time = max(times1_plot[-1], times2_plot[-1]) if len(times1_plot) > 0 and len(times2_plot) > 0 else 0
print(f"(0, {TARGET}) ({max_time:.2f}, {TARGET})")

if time1_target and time2_target:
    print(f"\nVertical lines (target reached):")
    print(f"% Baseline: x={time1_target:.2f}")
    print(f"({time1_target:.2f}, 0) ({time1_target:.2f}, {TARGET})")
    print(f"% Optimized: x={time2_target:.2f}")
    print(f"({time2_target:.2f}, 0) ({time2_target:.2f}, {TARGET})")
    
    print(f"\nSpeedup annotation:")
    print(f"% Arrow from ({time1_target:.2f}, {mid_y:.2f}) to ({time2_target:.2f}, {mid_y:.2f})")
    print(f"% Text at ({mid_x:.2f}, {mid_y + y_range * 0.08:.2f}): '{speedup_text}'")

print(f"\nNumber of data points:")
print(f"  - Baseline: {len(times1_plot)} points")
print(f"  - Optimized: {len(times2_plot)} points")
print(f"{'='*60}")

plt.show()