#!/usr/bin/env python3
"""
python plot_tta.py bsp.json sa.json --metric val_ppl --target 30 -o plot.pdf
  -> PDF plot + prints tex coordinates to stdout

python plot_tta.py bsp.json sa.json --metric val_ppl --target 30 --minival mini_val_ppl -o plot.pdf
  -> same but with minival points
"""
import json, argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

METRIC_LABELS = {
    "val_ppl": "Validation Perplexity",
    "val_loss": "Validation Loss",
    "train_ppl": "Training Perplexity",
    "train_loss": "Training Loss",
    "mini_val_ppl": "Validation Perplexity",
    "mini_val_loss": "Validation Loss",
}


def metric_label(m):
    return METRIC_LABELS.get(m, m.replace("_", r"\_"))


def extract_tta(data, metric, minival_metric=None, max_epochs=None):
    epochs = data["epochs"]
    minival = data.get("minival", {})
    cum = 0.0
    mins, vals = [], []
    epoch_keys = sorted(epochs, key=int)
    if max_epochs is not None:
        epoch_keys = epoch_keys[:max_epochs]
    for ek in epoch_keys:
        ep = epochs[ek]
        start = cum
        dur = ep["epoch_time"]
        steps = ep["steps"]
        if minival_metric and ek in minival:
            for sk in sorted(minival[ek], key=int):
                mins.append((start + (int(sk) / steps) * dur) / 60)
                vals.append(minival[ek][sk][minival_metric])
        cum += dur
        mins.append(cum / 60)
        vals.append(ep[metric])
    return np.array(mins), np.array(vals)


def find_crossing(m, v, target):
    for i in range(1, len(v)):
        if v[i] <= target and v[i - 1] > target:
            frac = (v[i - 1] - target) / (v[i - 1] - v[i])
            return m[i - 1] + frac * (m[i] - m[i - 1])
    return None


def print_tex_coordinates(series, crossings, labels, target):
    for i, (m, v, label) in enumerate(series):
        print(f"% {label}")
        coords = " ".join(f"({mi:.2f}, {vi:.2f})" for mi, vi in zip(m, v))
        print(coords)
        if target and crossings[i]:
            print(f"% {label} crosses {target} at {crossings[i]:.2f} min")
        print()
    if target and crossings[0] and crossings[1]:
        sp = max(crossings) / min(crossings)
        print(f"% Speedup: {sp:.2f}x")


def plot_tta(file_a, file_b, metric, labels=("BSP", "SA-INA"),
             minival_metric=None, target=None, output=None, smooth=False,
             title=None, epochs=None):
    import plotconfig
    import matplotlib.pyplot as plt

    with open(file_a) as f: da = json.load(f)
    with open(file_b) as f: db = json.load(f)

    # Auto-match epoch count if not specified
    if epochs is None:
        na = len(da["epochs"])
        nb = len(db["epochs"])
        if na != nb:
            epochs = min(na, nb)

    series = []
    crossings = []
    for data, label in [(da, labels[0]), (db, labels[1])]:
        m, v = extract_tta(data, metric, minival_metric, epochs)
        series.append((m, v, label))
        crossings.append(find_crossing(m, v, target) if target else None)

    # Print tex coordinates
    print_tex_coordinates(series, crossings, labels, target)

    # Plot
    fig, ax = plt.subplots(figsize=(plotconfig.COLUMN_WIDTH, 2.2))
    colors = ["#E63946", "#2A9D8F"]
    marks = ["s", "o"]

    for i, (m, v, label) in enumerate(series):
        if smooth and not minival_metric:
            from scipy.interpolate import PchipInterpolator
            interp = PchipInterpolator(m, v)
            ms = np.linspace(m[0], m[-1], 300)
            ax.plot(ms, interp(ms), color=colors[i], lw=1.2, label=label, zorder=3)
            ax.scatter(m, v, color=colors[i], s=14, zorder=4,
                       marker=marks[i], edgecolors="white", linewidths=0.4)
        else:
            ax.plot(m, v, color=colors[i], lw=1.2, label=label,
                    marker=marks[i], ms=3.5, markeredgecolor="white",
                    markeredgewidth=0.4, zorder=3)

    if target:
        ax.axhline(target, color="#4B5157", ls="--", lw=0.7, zorder=2)
        tc_min = min(tc for tc in crossings if tc)
        tc_max = max(tc for tc in crossings if tc)
        # Shaded region
        ax.axvspan(tc_min, tc_max, ymin=0,
                    ymax=(target - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    color="green", alpha=0.06, zorder=1)
        # Crossing verticals
        for i, tc in enumerate(crossings):
            if tc:
                ax.axvline(tc, color=colors[i], ls=":", lw=0.6, alpha=0.5, ymax=0.95, zorder=2)
        # Speedup arrow
        sp = tc_max / tc_min
        arrow_y = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.35
        ax.annotate("", xy=(tc_min, arrow_y), xytext=(tc_max, arrow_y),
                     arrowprops=dict(arrowstyle="<->", color="black", lw=0.6, alpha=0.4))
        ax.text((tc_min + tc_max) / 2, arrow_y, f"{sp:.2f}x",
                ha="center", va="center", fontsize=6,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="gray", lw=0.3, alpha=0.9),
                color="0.4", fontweight="bold")

    # Fixed-epoch comparison: vertical lines at last point of each curve + arrow
    end_times = [series[i][0][-1] for i in range(2)]
    end_vals = [series[i][1][-1] for i in range(2)]
    ylim = ax.get_ylim()
    # Dashed lines (distinct from the dotted crossing lines)
    for i in range(2):
        ax.axvline(end_times[i], color=colors[i], ls=(0, (5, 3)), lw=0.7, alpha=0.5, zorder=2)
    # Arrow positioned at 15% from top of plot
    arrow_y = ylim[1] - (ylim[1] - ylim[0]) * 0.15
    time_saved = abs(end_times[0] - end_times[1])
    ratio = max(end_times) / min(end_times)
    ax.annotate("", xy=(min(end_times), arrow_y), xytext=(max(end_times), arrow_y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=0.6, alpha=0.4))
    ax.text((end_times[0] + end_times[1]) / 2, arrow_y, f"{ratio:.2f}x",
            ha="center", va="center", fontsize=6,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="gray", lw=0.3, alpha=0.9),
            color="0.4", fontweight="bold")

    # Print fixed-epoch info
    print(f"\n% Fixed-epoch ({epochs or 'auto'}):")
    print(f"%   {labels[0]} ends at {end_times[0]:.0f} min, {metric}={end_vals[0]:.2f}")
    print(f"%   {labels[1]} ends at {end_times[1]:.0f} min, {metric}={end_vals[1]:.2f}")
    print(f"%   Time ratio: {ratio:.2f}x ({time_saved:.0f} min saved)")

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(metric_label(metric))
    if title:
        ax.set_title(title, fontweight="bold", fontfamily="sans-serif")
    ax.legend(loc="upper right")
    fig.tight_layout()

    if output:
        fig.savefig(output)
        print(f"\nSaved {output}")
    else:
        plt.show()
    plt.close(fig)



FILE_SU="data/train/gpt2/moderate/sa/rank0.json"
FILE_SA="data/train/gpt2/moderate/sa/rank0.json" #"data/train/gpt2/aggressive/sa/rank0.json"

plot_tta(FILE_SU, FILE_SA, "val_ppl", ("SU", "SA"), "mini_val_ppl", 28, "gpt2_ttm.pdf", 0)


# if __name__ == "__main__":
#     p = argparse.ArgumentParser()
#     p.add_argument("file_a", help="Baseline (red)")
#     p.add_argument("file_b", help="Optimized (teal)")
#     p.add_argument("--metric", default="val_ppl")
#     p.add_argument("--minival", default=None)
#     p.add_argument("--labels", nargs=2, default=["BSP", "SA-INA"])
#     p.add_argument("--target", type=float, default=None)
#     p.add_argument("--smooth", action="store_true")
#     p.add_argument("--title", default=None)
#     p.add_argument("--epochs", type=int, default=None, help="Max epochs to show (auto-matches if unset)")
#     p.add_argument("-o", "--output", default=None)
#     a = p.parse_args()
#     plot_tta(a.file_a, a.file_b, a.metric, a.labels, a.minival, a.target, a.output, a.smooth, a.title, a.epochs)

