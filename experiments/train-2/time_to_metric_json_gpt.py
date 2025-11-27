import json, os
import matplotlib.pyplot as plt
import numpy as np

# File paths with clearer names
FILE_NO_STRAGGLE = '../train/results/gpt2.json'               # no stragglers
FILE_STRAGGLE_UNAWARE = '../train/results/gpt2_straggle_16.json'   # with stragglers, no mitigation
FILE_STRAGGLE_AWARE = '../train-2/gpt2_straggle_16.json'         # with stragglers and mitigation

FILE_NO_STRAGGLE = os.path.join(os.path.dirname(__file__), FILE_NO_STRAGGLE)
FILE_STRAGGLE_UNAWARE = os.path.join(os.path.dirname(__file__), FILE_STRAGGLE_UNAWARE)
FILE_STRAGGLE_AWARE = os.path.join(os.path.dirname(__file__), FILE_STRAGGLE_AWARE)

# Configuration
METRIC = "val_ppl"  # Change to match your data: val_ppl, val_loss, etc.
MINIVAL_METRIC = "mini_val_ppl"  # The corresponding minival metric
YLABEL = "Perplexity"  
TARGET = 30  # Target value
TARGET_TOL = 2.5
LOWER_IS_BETTER = False  # True for perplexity, loss; False for accuracy
USE_MINIVAL = True  # Set to False to only use epoch data
USE_MINIVAL_EPOCHS = 3

# Load data
with open(FILE_NO_STRAGGLE, 'r') as f: 
    data_no_straggle = json.loads(f.read())
with open(FILE_STRAGGLE_UNAWARE, 'r') as f: 
    data_straggle_unaware = json.loads(f.read())
with open(FILE_STRAGGLE_AWARE, 'r') as f: 
    data_straggle_aware = json.loads(f.read())

# Debug: Print step times for each epoch
def print_epoch_step_times(data_name, epochs_data):
    print(f"\n{data_name} - Step times per epoch:")
    for epoch_key in sorted(epochs_data.keys(), key=int):
        epoch_data = epochs_data[epoch_key]
        print(f"  Epoch {epoch_key}: {epoch_data['step_time']:.4f} sec/step ({epoch_data['steps']} steps)")

print(f"\n{'='*60}")
print(f"EPOCH STEP TIMES")
print(f"{'='*60}")
print_epoch_step_times("No Stragglers", data_no_straggle['epochs'])
print_epoch_step_times("Straggler Unaware", data_straggle_unaware['epochs'])
print_epoch_step_times("Straggler Aware", data_straggle_aware['epochs'])

def extract_combined_data_step_based(epochs_data, minival_data=None, use_minival=True):
    """Extract both epoch and minival data points with proper time accumulation"""
    all_points = []  # List of (global_step, metric, cumulative_time) tuples
    
    # Build a map of epoch info with cumulative timing
    epoch_info = {}
    cumulative_steps = 0
    cumulative_time = 0.0
    
    for epoch_key in sorted(epochs_data.keys(), key=int):
        epoch_num = int(epoch_key)
        epoch_data = epochs_data[epoch_key]
        steps_in_epoch = epoch_data['steps']
        step_time = epoch_data['step_time']
        
        epoch_info[epoch_num] = {
            'start_steps': cumulative_steps,
            'end_steps': cumulative_steps + steps_in_epoch,
            'start_time': cumulative_time,
            'end_time': cumulative_time + (steps_in_epoch * step_time),
            'step_time': step_time,
            'steps': steps_in_epoch
        }
        
        cumulative_steps += steps_in_epoch
        cumulative_time += steps_in_epoch * step_time
    
    # Add minival points if requested and available
    if use_minival and minival_data:
        for epoch_key in sorted(minival_data.keys(), key=int):
            epoch_num = int(epoch_key)
            if USE_MINIVAL_EPOCHS and epoch_num >= USE_MINIVAL_EPOCHS:
                break
            if epoch_num not in epoch_info:
                continue
                
            epoch_minival = minival_data[epoch_key]
            einfo = epoch_info[epoch_num]
            
            # Add each minival checkpoint
            for step_key in sorted(epoch_minival.keys()):
                step_data = epoch_minival[step_key]
                global_step = step_data['global_step']
                metric_value = step_data[MINIVAL_METRIC]
                
                # Calculate cumulative time up to this global step
                # First add time from all complete previous epochs
                time_so_far = 0
                for e in range(epoch_num):
                    if e in epoch_info:
                        time_so_far += (epoch_info[e]['steps'] * epoch_info[e]['step_time'])
                
                # Then add time for steps within current epoch
                steps_in_current_epoch = global_step - einfo['start_steps']
                time_so_far += steps_in_current_epoch * einfo['step_time']
                
                all_points.append((global_step, metric_value, time_so_far))
    
    # Add epoch-end points
    for epoch_key in sorted(epochs_data.keys(), key=int):
        epoch_num = int(epoch_key)
        epoch_data = epochs_data[epoch_key]
        
        if epoch_num not in epoch_info:
            continue
            
        einfo = epoch_info[epoch_num]
        
        # Get global step - prefer explicit global_step if available
        if 'global_step' in epoch_data:
            global_step = epoch_data['global_step']
        else:
            global_step = einfo['end_steps']
        
        metric_value = epoch_data[METRIC]
        time_at_epoch_end = einfo['end_time']
        
        # Only add if not a duplicate of last minival point
        if not all_points or all_points[-1][0] != global_step:
            all_points.append((global_step, metric_value, time_at_epoch_end))
    
    # Sort by global_step and extract arrays
    all_points.sort(key=lambda x: x[0])
    global_steps = [p[0] for p in all_points]
    metrics = [p[1] for p in all_points]
    times = [p[2] for p in all_points]
    
    return global_steps, metrics, times

def extract_epoch_only_data_step_based(epochs_data):
    """Extract only epoch-level data with proper time accumulation"""
    epoch_numbers = []
    epoch_metrics = []
    cumulative_times = []
    global_steps = []
    
    cumulative_time = 0.0
    cumulative_steps = 0
    
    for epoch_key in sorted(epochs_data.keys(), key=int):
        epoch_num = int(epoch_key)
        epoch_data = epochs_data[epoch_key]
        
        steps_in_epoch = epoch_data['steps']
        step_time = epoch_data['step_time']
        
        # Add time for this epoch
        cumulative_time += steps_in_epoch * step_time
        cumulative_steps += steps_in_epoch
        
        # Get global step - prefer explicit global_step if available
        if 'global_step' in epoch_data:
            global_step = epoch_data['global_step']
        else:
            global_step = cumulative_steps
        
        # Store data
        epoch_numbers.append(epoch_num)
        epoch_metrics.append(epoch_data[METRIC])
        cumulative_times.append(cumulative_time)
        global_steps.append(global_step)
    
    return global_steps, epoch_metrics, cumulative_times

# Extract data for all three datasets
minival_no_straggle = data_no_straggle.get('minival', None) if USE_MINIVAL else None
minival_straggle_unaware = data_straggle_unaware.get('minival', None) if USE_MINIVAL else None
minival_straggle_aware = data_straggle_aware.get('minival', None) if USE_MINIVAL else None

if minival_no_straggle or minival_straggle_unaware or minival_straggle_aware:
    steps_no_straggle, metrics_no_straggle, times_no_straggle = \
        extract_combined_data_step_based(data_no_straggle['epochs'], minival_no_straggle, USE_MINIVAL)
    steps_straggle_unaware, metrics_straggle_unaware, times_straggle_unaware = \
        extract_combined_data_step_based(data_straggle_unaware['epochs'], minival_straggle_unaware, USE_MINIVAL)
    steps_straggle_aware, metrics_straggle_aware, times_straggle_aware = \
        extract_combined_data_step_based(data_straggle_aware['epochs'], minival_straggle_aware, USE_MINIVAL)
else:
    # Fallback to epoch-only data
    steps_no_straggle, metrics_no_straggle, times_no_straggle = \
        extract_epoch_only_data_step_based(data_no_straggle['epochs'])
    steps_straggle_unaware, metrics_straggle_unaware, times_straggle_unaware = \
        extract_epoch_only_data_step_based(data_straggle_unaware['epochs'])
    steps_straggle_aware, metrics_straggle_aware, times_straggle_aware = \
        extract_epoch_only_data_step_based(data_straggle_aware['epochs'])

# Convert to minutes
times_no_straggle_min = np.array(times_no_straggle) / 60
times_straggle_unaware_min = np.array(times_straggle_unaware) / 60
times_straggle_aware_min = np.array(times_straggle_aware) / 60

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

time_no_straggle_target, step_no_straggle_target, idx_no_straggle_target = \
    find_metric_target(metrics_no_straggle, TARGET, times_no_straggle_min, steps_no_straggle, LOWER_IS_BETTER)
time_straggle_unaware_target, step_straggle_unaware_target, idx_straggle_unaware_target = \
    find_metric_target(metrics_straggle_unaware, TARGET, times_straggle_unaware_min, steps_straggle_unaware, LOWER_IS_BETTER)
time_straggle_aware_target, step_straggle_aware_target, idx_straggle_aware_target = \
    find_metric_target(metrics_straggle_aware, TARGET, times_straggle_aware_min, steps_straggle_aware, LOWER_IS_BETTER)

# Truncate data to show only a few points after hitting target
def truncate_after_target(times, metrics, idx_target, extra_points=5):
    if idx_target is not None:
        end_idx = min(idx_target + extra_points, len(times))
        return times[:end_idx], metrics[:end_idx]
    return times, metrics

times_no_straggle_plot, metrics_no_straggle_plot = \
    truncate_after_target(times_no_straggle_min, metrics_no_straggle, idx_no_straggle_target, 8)
times_straggle_unaware_plot, metrics_straggle_unaware_plot = \
    truncate_after_target(times_straggle_unaware_min, metrics_straggle_unaware, idx_straggle_unaware_target, 8)
times_straggle_aware_plot, metrics_straggle_aware_plot = \
    truncate_after_target(times_straggle_aware_min, metrics_straggle_aware, idx_straggle_aware_target, 8)

# Create the plot
plt.figure(figsize=(12, 7))

# Plot all three lines
marker_size = 4 if USE_MINIVAL else 5
plt.plot(times_no_straggle_plot, metrics_no_straggle_plot, 
         marker='s', linewidth=2.5, markersize=marker_size, 
         color="#67A25A", label='No Stragglers', alpha=0.9)
plt.plot(times_straggle_unaware_plot, metrics_straggle_unaware_plot, 
         marker='o', linewidth=2.5, markersize=marker_size, 
         color='#D62828', label='Straggler Unaware', alpha=0.9)
plt.plot(times_straggle_aware_plot, metrics_straggle_aware_plot, 
         marker='^', linewidth=2.5, markersize=marker_size, 
         color='#1E88E5', label='Straggler Aware', alpha=0.9)

# Add horizontal line at target
# target_label = f'Target {"≤" if LOWER_IS_BETTER else "≥"} {TARGET}'
# plt.axhline(y=TARGET, color='gray', linestyle='--', linewidth=1.5, label=target_label, alpha=0.7)

# Add vertical lines when each hits target
y_bottom = plt.ylim()[0]
# if time_no_straggle_target:
#     plt.axvline(x=time_no_straggle_target, color='#67A25A', linestyle=':', linewidth=1.5, alpha=0.6)
#     plt.text(time_no_straggle_target, y_bottom + (TARGET - y_bottom) * 0.05, 
#              f'{time_no_straggle_target:.1f} min', rotation=90, {'linestyle': '-', 'marker': 'X', 'linewidth': 0.5,  'markersize': 6 },
#              verticalalignment='bottom', horizontalalignment='right', 
#              color='#67A25A', fontsize=9, fontweight='bold')

# if time_straggle_unaware_target:
#     plt.axvline(x=time_straggle_unaware_target, color='#D62828', linestyle=':', linewidth=1.5, alpha=0.6)
#     plt.text(time_straggle_unaware_target, y_bottom + (TARGET - y_bottom) * 0.05, 
#              f'{time_straggle_unaware_target:.1f} min', rotation=90, 
#              verticalalignment='bottom', horizontalalignment='right', 
#              color='#D62828', fontsize=9, fontweight='bold')

# if time_straggle_aware_target:
#     plt.axvline(x=time_straggle_aware_target, color='#1E88E5', linestyle=':', linewidth=1.5, alpha=0.6)
#     plt.text(time_straggle_aware_target, y_bottom + (TARGET - y_bottom) * 0.05, 
#              f'{time_straggle_aware_target:.1f} min', rotation=90, 
#              verticalalignment='bottom', horizontalalignment='right', 
#              color='#1E88E5', fontsize=9, fontweight='bold')

# Calculate speedup between straggle unaware and straggle aware
speedup_text = ""
if time_straggle_unaware_target and time_straggle_aware_target:
    speedup = time_straggle_unaware_target / time_straggle_aware_target
    speedup_text = f'{speedup:.2f}x speedup'
    
    # Add speedup annotation
    mid_x = (time_straggle_unaware_target + time_straggle_aware_target) / 2
    
    # Position at middle of plot
    y_range = max(max(metrics_straggle_unaware_plot), max(metrics_straggle_aware_plot)) - \
              min(min(metrics_straggle_unaware_plot), min(metrics_straggle_aware_plot))
    y_min_plot = min(min(metrics_straggle_unaware_plot), min(metrics_straggle_aware_plot))
    mid_y = y_min_plot + y_range * 0.5
    
    plt.annotate('', xy=(time_straggle_aware_target, mid_y), 
                 xytext=(time_straggle_unaware_target, mid_y), 
                 arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    plt.text(mid_x, mid_y + y_range * 0.08, speedup_text, 
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFA500', 
                      edgecolor='black', linewidth=1.5, alpha=0.85))

# Formatting
plt.xlabel('Time (minutes)', fontsize=12, fontweight='bold')
plt.ylabel(f'{YLABEL}', fontsize=12, fontweight='bold')

# Title includes whether we're using minival data
data_type = "with Intermediate Checkpoints" if USE_MINIVAL and \
            (minival_straggle_unaware or minival_straggle_aware) else ""
direction_text = "Lower" if LOWER_IS_BETTER else "Higher"
plt.title(f'Training Progress Comparison: Straggler Impact and Mitigation\n{data_type} ({direction_text} is Better)', 
          fontsize=14, fontweight='bold', pad=20)

plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=10, loc='best')

# Set y-axis limits based on data
all_metrics = metrics_no_straggle_plot + metrics_straggle_unaware_plot + metrics_straggle_aware_plot
if LOWER_IS_BETTER:
    y_min = min(min(all_metrics), TARGET) * 0.95
    y_max = max(all_metrics[0:3]) * 1.05  # Use first few points for max
    plt.ylim(y_min, y_max)
else:
    y_min = max(0, min(all_metrics) * 0.95)
    y_max = min(100, max(all_metrics) * 1.05)
    plt.ylim(y_min, y_max)

plt.tight_layout()

# Save the plot (uncomment if needed)
# plt.savefig('/mnt/user-data/outputs/time_to_metric_comparison.png', dpi=300, bbox_inches='tight')
# print(f"Plot saved to /mnt/user-data/outputs/time_to_metric_comparison.png")

# Print statistics
print(f"\n{'='*60}")
print(f"TIME-TO-METRIC COMPARISON")
print(f"{'='*60}")

if time_no_straggle_target:
    print(f"\nNo Stragglers:")
    print(f"  - Reached {TARGET} at step {step_no_straggle_target}: {time_no_straggle_target:.2f} minutes")
    print(f"  - Final {YLABEL}: {metrics_no_straggle[-1]:.2f}")
    print(f"  - Total time: {times_no_straggle_min[-1]:.2f} minutes")
    print(f"  - Total steps: {steps_no_straggle[-1]}")

if time_straggle_unaware_target:
    print(f"\nStraggler Unaware:")
    print(f"  - Reached {TARGET} at step {step_straggle_unaware_target}: {time_straggle_unaware_target:.2f} minutes")
    print(f"  - Final {YLABEL}: {metrics_straggle_unaware[-1]:.2f}")
    print(f"  - Total time: {times_straggle_unaware_min[-1]:.2f} minutes")
    print(f"  - Total steps: {steps_straggle_unaware[-1]}")

if time_straggle_aware_target:
    print(f"\nStraggler Aware:")
    print(f"  - Reached {TARGET} at step {step_straggle_aware_target}: {time_straggle_aware_target:.2f} minutes")
    print(f"  - Final {YLABEL}: {metrics_straggle_aware[-1]:.2f}")
    print(f"  - Total time: {times_straggle_aware_min[-1]:.2f} minutes")
    print(f"  - Total steps: {steps_straggle_aware[-1]}")

if time_straggle_unaware_target and time_straggle_aware_target:
    print(f"\n{'='*60}")
    print(f"SPEEDUP RESULTS (Straggle Aware vs Unaware)")
    print(f"{'='*60}")
    time_saved = time_straggle_unaware_target - time_straggle_aware_target
    speedup = time_straggle_unaware_target / time_straggle_aware_target
    print(f"  - Time saved to {TARGET}: {time_saved:.2f} minutes ({time_saved/60:.2f} hours)")
    print(f"  - Speedup factor: {speedup:.2f}x")
    print(f"  - Percentage improvement: {(1 - 1/speedup)*100:.1f}%")
    print(f"{'='*60}")

# Print coordinates for LaTeX
print(f"\n{'='*60}")
print(f"PLOT COORDINATES FOR LATEX")
print(f"{'='*60}")

print(f"\n% No Stragglers")
for t, m in zip(times_no_straggle_plot, metrics_no_straggle_plot):
    print(f"({t:.2f}, {m:.2f})")

print(f"\n% Straggler Unaware")
for t, m in zip(times_straggle_unaware_plot, metrics_straggle_unaware_plot):
    print(f"({t:.2f}, {m:.2f})")

print(f"\n% Straggler Aware")
for t, m in zip(times_straggle_aware_plot, metrics_straggle_aware_plot):
    print(f"({t:.2f}, {m:.2f})")

print(f"{'='*60}")

plt.show()