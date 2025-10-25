import json, os
import matplotlib.pyplot as plt
import numpy as np

FILE1='../experiments/train/results/deterministic/resnet50.json'
FILE2='../experiments/train/results/deterministic/resnet50_straggle_16.json'
FILE1=os.path.join(os.path.dirname(__file__), FILE1)
FILE2=os.path.join(os.path.dirname(__file__), FILE2)

# Read the first dataset (baseline)
with open(FILE1, 'r') as f:
    # content = f.read()
    # data1 = json.loads('{' + content + '}')
    data1 = json.loads(f.read())

# Read the second dataset
with open(FILE2, 'r') as f:
    # content = f.read()
    # data2 = json.loads('{' + content + '}')
    data2 = json.loads(f.read())

def extract_data(epochs_data):
    """Extract epoch numbers, accuracies, and cumulative times from epoch data"""
    epoch_numbers = []
    val_top5_accuracies = []
    cumulative_times = []
    
    cumulative_time = 0.0
    for epoch_key in sorted(epochs_data.keys(), key=int):
        epoch_data = epochs_data[epoch_key]
        
        # Calculate time for this epoch: steps * step_time
        steps = epoch_data['steps']
        step_time = epoch_data['step_time']
        epoch_time = steps * step_time
        cumulative_time += epoch_time
        
        # Store data
        epoch_numbers.append(int(epoch_key))
        val_top5_accuracies.append(epoch_data['val_top5'])
        cumulative_times.append(cumulative_time)
    
    return epoch_numbers, val_top5_accuracies, cumulative_times

# Extract data for both datasets
epochs1, acc1, times1 = extract_data(data1['epochs'])
epochs2, acc2, times2 = extract_data(data2['epochs'])

# Convert to minutes
times1_min = np.array(times1) / 60
times2_min = np.array(times2) / 60

# Find when each hits 90% accuracy
def find_90_percent(accuracies, times_min, epochs):
    for i, acc in enumerate(accuracies):
        if acc >= 90.0:
            return times_min[i], epochs[i], i
    return None, None, None

time1_90, epoch1_90, idx1_90 = find_90_percent(acc1, times1_min, epochs1)
time2_90, epoch2_90, idx2_90 = find_90_percent(acc2, times2_min, epochs2)

# Truncate data to show only 2 points after hitting 90%
def truncate_after_target(times, accs, idx_90):
    if idx_90 is not None:
        end_idx = min(idx_90 + 3, len(times))  # Show 2 more points after target
        return times[:end_idx], accs[:end_idx]
    return times, accs

times1_plot, acc1_plot = truncate_after_target(times1_min, acc1, idx1_90)
times2_plot, acc2_plot = truncate_after_target(times2_min, acc2, idx2_90)

# Calculate speedup
if time1_90 and time2_90:
    speedup = time1_90 / time2_90
    time_saved = time1_90 - time2_90

# Create the plot
plt.figure(figsize=(12, 7))

# Plot both lines with thicker lines, red and blue
plt.plot(times1_plot, acc1_plot, marker='o', linewidth=3, markersize=5, 
         color='#D62828', label='Baseline', alpha=0.9)
plt.plot(times2_plot, acc2_plot, marker='s', linewidth=3, markersize=5, 
         color='#1E88E5', label='Optimized', alpha=0.9)

# Add horizontal line at 90% target accuracy
plt.axhline(y=90, color='gray', linestyle='--', linewidth=1.5, 
            label='Target Accuracy (90%)', alpha=0.7)

# Add vertical lines when each hits 90% - stop at the 90% horizontal line
if time1_90:
    plt.axvline(x=time1_90, color='#D62828', linestyle=':', linewidth=1.5, alpha=0.6, ymax=0.9)
    plt.text(time1_90, 5, f'{time1_90:.1f} min', 
             rotation=90, verticalalignment='bottom', horizontalalignment='right',
             color='#D62828', fontsize=9, fontweight='bold')

if time2_90:
    plt.axvline(x=time2_90, color='#1E88E5', linestyle=':', linewidth=1.5, alpha=0.6, ymax=0.9)
    plt.text(time2_90, 5, f'{time2_90:.1f} min', 
             rotation=90, verticalalignment='bottom', horizontalalignment='right',
             color='#1E88E5', fontsize=9, fontweight='bold')

# Add speedup annotation in the middle of y-axis
if time1_90 and time2_90:
    mid_x = (time1_90 + time2_90) / 2
    mid_y = 50  # Middle of y-axis
    plt.annotate('', xy=(time2_90, mid_y), xytext=(time1_90, mid_y),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    plt.text(mid_x, mid_y + 4, f'{speedup:.2f}x faster', 
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFA500', edgecolor='black', 
                      linewidth=1.5, alpha=0.85))

# Formatting
plt.xlabel('Time (minutes)', fontsize=12, fontweight='bold')
plt.ylabel('Top-5 Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Time-to-Accuracy Comparison: Baseline vs Optimized', 
          fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=10, loc='lower left')

# Set y-axis limits for better visualization
plt.ylim(0, 100)

# Add tight layout
plt.tight_layout()

# Save the plot
# plt.savefig('/mnt/user-data/outputs/time_to_accuracy_comparison.png', dpi=300, bbox_inches='tight')
# print(f"Plot saved to /mnt/user-data/outputs/time_to_accuracy_comparison.png")

# Print statistics
print(f"\n{'='*60}")
print(f"TIME-TO-ACCURACY COMPARISON")
print(f"{'='*60}")
print(f"\nBaseline:")
print(f"  - Reached 90% at epoch {epoch1_90}: {time1_90:.2f} minutes")
print(f"  - Final accuracy: {acc1[-1]:.2f}%")
print(f"  - Total time: {times1_min[-1]:.2f} minutes")

print(f"\nOptimized:")
print(f"  - Reached 90% at epoch {epoch2_90}: {time2_90:.2f} minutes")
print(f"  - Final accuracy: {acc2[-1]:.2f}%")
print(f"  - Total time: {times2_min[-1]:.2f} minutes")

print(f"\n{'='*60}")
print(f"SPEEDUP RESULTS")
print(f"{'='*60}")
print(f"  - Time saved to 90%: {time_saved:.2f} minutes ({time_saved/60:.2f} hours)")
print(f"  - Speedup factor: {speedup:.2f}x")
print(f"  - Percentage improvement: {(1 - 1/speedup)*100:.1f}%")
print(f"{'='*60}")

# Print coordinates for LaTeX
print(f"\n{'='*60}")
print(f"PLOT COORDINATES (for LaTeX)")
print(f"{'='*60}")
print(f"\nBaseline coordinates:")
print("% Time (min), Accuracy (%)")
for t, a in zip(times1_plot, acc1_plot):
    print(f"({t:.2f}, {a:.2f})")

print(f"\nOptimized coordinates:")
print("% Time (min), Accuracy (%)")
for t, a in zip(times2_plot, acc2_plot):
    print(f"({t:.2f}, {a:.2f})")

print(f"\nTarget line:")
print(f"% Horizontal line at y=90")
print(f"(0, 90) ({max(times1_plot[-1], times2_plot[-1]):.2f}, 90)")

print(f"\nVertical lines (90% achievement):")
print(f"% Baseline: x={time1_90:.2f}")
print(f"% Optimized: x={time2_90:.2f}")
print(f"{'='*60}")


plt.show()