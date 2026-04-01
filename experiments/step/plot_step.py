import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Replace this with your actual populated JSON/Dictionary data
data = {
  "sa" : {
    "gpt": {"0": {"avg": [105]}, "5": {"avg": [100]}, "10": {"avg": [95]}, "15": {"avg": [85]}, "20": {"avg": [75]}},
    "qwen": {"0": {"avg": [120]}, "5": {"avg": [115]}, "10": {"avg": [105]}, "15": {"avg": [95]}, "20": {"avg": [80]}},
    "roberta": {"0": {"avg": [80]}, "5": {"avg": [75]}, "10": {"avg": [70]}, "15": {"avg": [60]}, "20": {"avg": [55]}},
    "resnet": {"0": {"avg": [50]}, "5": {"avg": [48]}, "10": {"avg": [45]}, "15": {"avg": [40]}, "20": {"avg": [38]}}
  },
  "su" : {
    "gpt": {"0": {"avg": [120]}, "5": {"avg": [130]}, "10": {"avg": [140]}, "15": {"avg": [150]}, "20": {"avg": [160]}},
    "qwen": {"0": {"avg": [130]}, "5": {"avg": [140]}, "10": {"avg": [150]}, "15": {"avg": [165]}, "20": {"avg": [180]}},
    "roberta": {"0": {"avg": [85]}, "5": {"avg": [90]}, "10": {"avg": [95]}, "15": {"avg": [105]}, "20": {"avg": [115]}},
    "resnet": {"0": {"avg": [55]}, "5": {"avg": [58]}, "10": {"avg": [62]}, "15": {"avg": [68]}, "20": {"avg": [75]}}
  },
  "ideal" : {
    "gpt": {"avg": [90]},    # These are dummy 'ideal' times 
    "qwen": {"avg": [100]},
    "roberta": {"avg": [65]},
    "resnet": {"avg": [40]}
  }
}

DIR=Path(__file__).parent
with open(DIR / "step.json") as f:
    data = json.load(f)



models = ["resnet", "gpt", "roberta", "qwen"]
probabilities = ["0", "5", "10", "15", "20"]  # These represent your thresholds/probabilities

# Create a 2x2 grid for our subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Ensure probabilities are sorted numerically for the x-axis
x_labels = sorted([int(p) for p in probabilities])

data = data["3"]
for idx, model in enumerate(models):
    ax = axes[idx]
    
    # 1. Get the ideal time (the denominator for normalization)
    ideal_time = np.mean(data["ideal"][model]["avg"])
    
    # 2. Extract SA and SU average times, then divide by ideal_time
    sa_time_norm = [np.mean(data["sa"][model][str(x)]["avg"]) / ideal_time for x in x_labels]
    su_time_norm = [np.mean(data["su"][model][str(x)]["avg"]) / ideal_time for x in x_labels]
    
    # The ideal baseline normalized against itself is exactly 1.0
    ideal_baseline = 1.0
    
    # 3. Plot SA and SU normalized times
    ax.plot(x_labels, sa_time_norm, marker='o', label='SA Time', color='#1f77b4', linewidth=2)
    ax.plot(x_labels, su_time_norm, marker='s', label='SU Time', color='#ff7f0e', linewidth=2)
    
    # 4. Plot the Ideal horizontal line at 1.0
    ax.axhline(y=ideal_baseline, color='#d62728', linestyle='--', label='Ideal (1.0)', linewidth=2)
    
    # 5. Format the subplot
    ax.set_title(f'Model: {model.upper()}', fontsize=14, pad=10)
    ax.set_xlabel('Probabilities (%)', fontsize=12)
    ax.set_ylabel('Normalized Time (Relative to Ideal)', fontsize=12)
    ax.set_xticks(x_labels) # Ensure the x-axis ticks align exactly with the probability increments
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='best')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('step.pdf')


plt.show()



for model in models:
    print(f"% ==========================================")
    print(f"% Coordinates for Model: {model.upper()}")
    print(f"% ==========================================\n")
    
    # Calculate the ideal time (denominator for normalization)
    ideal_time = np.mean(data["ideal"][model]["avg"])
    
    # Lists to hold the coordinate strings
    sa_coords = []
    su_coords = []
    
    for x in probabilities:
        # Calculate SA and SU averages, then normalize
        sa_val = np.mean(data["sa"][model][str(x)]["avg"]) / ideal_time
        su_val = np.mean(data["su"][model][str(x)]["avg"]) / ideal_time
        
        # Format as (x, y) rounding to 4 decimal places for LaTeX
        sa_coords.append(f"({x}, {sa_val:.4f})")
        su_coords.append(f"({x}, {su_val:.4f})")
    
    # Print SA coordinates
    print(f"% SA Line ({model.upper()})")
    print(f"\\addplot coordinates {{ {' '.join(sa_coords)} }};")
    print(f"\\addlegendentry{{SA Time}}\n")
    
    # Print SU coordinates
    print(f"% SU Line ({model.upper()})")
    print(f"\\addplot coordinates {{ {' '.join(su_coords)} }};")
    print(f"\\addlegendentry{{SU Time}}\n")
    
    # Print Ideal coordinates (Horizontal line at y=1.0)
    # We only need the first and last x-values to draw a straight line
    print(f"% Ideal Baseline ({model.upper()})")
    print(f"\\addplot[red, dashed] coordinates {{ ({probabilities[0]}, 1.0000) ({probabilities[-1]}, 1.0000) }};")
    print(f"\\addlegendentry{{Ideal Baseline}}\n")