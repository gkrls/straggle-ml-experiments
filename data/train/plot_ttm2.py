#!/usr/bin/env python3
"""
python plot_tta.py data/train/gpt2/moderate/su/ data/train/gpt2/moderate/sa/ \
    --metric val_ppl --target 28 --rank-agg avg --exclude-ranks 3 -o plot.pdf

Rank aggregation modes:
  rankN   - use a specific rank (e.g. rank0)
  avg     - average metric across ranks, max time (wall-clock)
  best    - best metric per epoch across ranks (min for loss, max for acc)
"""
import json, argparse, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

METRIC_LABELS = {
    "val_ppl":       "Validation Perplexity",
    "val_loss":      "Validation Loss",
    "train_ppl":     "Training Perplexity",
    "train_loss":    "Training Loss",
    "mini_val_ppl":  "Validation Perplexity",
    "mini_val_loss": "Validation Loss",
    "top1_acc":      "Top-1 Accuracy (\\%)",
    "top5_acc":      "Top-5 Accuracy (\\%)",
    "f1":            "F1 Score",
    "precision":     "Precision",
    "recall":        "Recall",
    "val_acc":       "Validation Accuracy (\\%)",
    "train_acc":     "Training Accuracy (\\%)",
    "val_top1":      "Top-1 Accuracy (\\%)",
    "val_top5":      "Top-5 Accuracy (\\%)",
    "train_top1":    "Top-1 Accuracy (\\%)",
    "train_top5":    "Top-5 Accuracy (\\%)",
    "exact_match":   "Exact Match",
}

LOWER_IS_BETTER = {"loss", "ppl"}


def metric_label(m):
    return METRIC_LABELS.get(m, m.replace("_", r"\_"))


def higher_is_better(metric):
    return not any(k in metric for k in LOWER_IS_BETTER)


# ---------------------------------------------------------------------------
# Rank loading & aggregation
# ---------------------------------------------------------------------------

def discover_ranks(directory, exclude_ranks=None):
    """Find rank*.json files in directory, return sorted (rank_id, path) list."""
    exclude = set(exclude_ranks or [])
    pairs = []
    for path in sorted(glob.glob(os.path.join(directory, "rank*.json"))):
        fname = os.path.basename(path)
        rank_id = int(fname.replace("rank", "").replace(".json", ""))
        if rank_id not in exclude:
            pairs.append((rank_id, path))
    if not pairs:
        raise FileNotFoundError(f"No rank*.json files in {directory} "
                                f"(excluded: {exclude})")
    return pairs


def load_ranks(directory, exclude_ranks=None):
    """Load all rank JSON files from a directory."""
    pairs = discover_ranks(directory, exclude_ranks)
    print(f"% Loading {len(pairs)} ranks from {directory}: "
          f"{[r for r, _ in pairs]}")
    datas = {}
    for rank_id, path in pairs:
        with open(path) as f:
            datas[rank_id] = json.load(f)
    return datas


def aggregate_data(rank_datas, metric, agg, minival_metric=None):
    """Build a synthetic data dict aggregated across ranks.

    Time:   first rank's epoch times (non-straggler ranks are in sync).
    Metric: avg or best across ranks.
    """
    hib = higher_is_better(metric)
    rank_ids = sorted(rank_datas.keys())
    ref_id = rank_ids[0]

    # Find common epoch range across all ranks
    epoch_keys = sorted(rank_datas[ref_id]["epochs"], key=int)
    for rid in rank_ids[1:]:
        rkeys = sorted(rank_datas[rid]["epochs"], key=int)
        epoch_keys = epoch_keys[:min(len(epoch_keys), len(rkeys))]

    agg_epochs = {}
    agg_minival = {}

    for ek in epoch_keys:
        vals = []
        for rid in rank_ids:
            vals.append(rank_datas[rid]["epochs"][ek][metric])

        # Time from reference rank (non-stragglers are in sync)
        ref_ep = rank_datas[ref_id]["epochs"][ek]
        agg_time = ref_ep["epoch_time"]
        agg_steps = ref_ep["steps"]

        if agg == "avg":
            agg_val = np.mean(vals)
        elif agg == "best":
            agg_val = max(vals) if hib else min(vals)
        else:
            raise ValueError(f"Unknown aggregation: {agg}")

        agg_epochs[ek] = {
            "epoch_time": agg_time,
            "steps": agg_steps,
            metric: float(agg_val),
        }

        # Aggregate minival
        if minival_metric:
            all_steps = set()
            for rid in rank_ids:
                mv = rank_datas[rid].get("minival", {})
                if ek in mv:
                    all_steps.update(mv[ek].keys())
            if all_steps:
                agg_minival[ek] = {}
                for sk in sorted(all_steps, key=int):
                    mv_vals = []
                    for rid in rank_ids:
                        mv = rank_datas[rid].get("minival", {})
                        if ek in mv and sk in mv[ek]:
                            mv_vals.append(mv[ek][sk][minival_metric])
                    if mv_vals:
                        if agg == "avg":
                            mv_agg = np.mean(mv_vals)
                        elif agg == "best":
                            mv_hib = higher_is_better(minival_metric)
                            mv_agg = max(mv_vals) if mv_hib else min(mv_vals)
                        agg_minival[ek][sk] = {minival_metric: float(mv_agg)}

    result = {"epochs": agg_epochs}
    if agg_minival:
        result["minival"] = agg_minival
    return result


def find_fastest_rank(rank_datas, metric, minival_metric=None,
                      target=None, tolerance=0.0):
    """Find the rank whose curve crosses the target earliest.

    Returns that rank's raw data dict (unmodified).
    Falls back to rank with best final metric if no target or no crossing.
    """
    hib = higher_is_better(metric)
    rank_ids = sorted(rank_datas.keys())

    best_time = None
    best_rid = None

    for rid in rank_ids:
        m, v = extract_tta(rank_datas[rid], metric, minival_metric)
        if target is not None:
            tc = find_crossing(m, v, target, hib, tolerance)
            if tc is not None and (best_time is None or tc < best_time):
                best_time = tc
                best_rid = rid

    # Fallback: no rank crosses target — pick best final metric
    if best_rid is None:
        final_vals = {}
        for rid in rank_ids:
            epochs = rank_datas[rid]["epochs"]
            last_ek = sorted(epochs, key=int)[-1]
            final_vals[rid] = epochs[last_ek][metric]
        if hib:
            best_rid = max(final_vals, key=final_vals.get)
        else:
            best_rid = min(final_vals, key=final_vals.get)
        print(f"% fastest: no rank crosses target, using rank {best_rid} "
              f"(best final {metric}={final_vals[best_rid]:.2f})")
    else:
        print(f"% fastest: rank {best_rid} crosses {target} at {best_time:.2f} min")

    return rank_datas[best_rid]


def load_input(path_or_dir, rank="rank0", exclude_ranks=None,
               metric=None, minival_metric=None,
               target=None, tolerance=0.0):
    """Load from a file or directory with rank aggregation.

    rank: 'rank0', 'rank1', ..., 'avg', 'best', 'fastest'
    """
    if os.path.isfile(path_or_dir):
        print(f"% Using file: {path_or_dir}")
        with open(path_or_dir) as f:
            return json.load(f)

    if not os.path.isdir(path_or_dir):
        raise FileNotFoundError(f"Not a file or directory: {path_or_dir}")

    # Normalize "0" -> "rank0"
    if rank.isdigit():
        rank = f"rank{rank}"

    # Specific rank
    if rank.startswith("rank"):
        rank_id = int(rank.replace("rank", ""))
        path = os.path.join(path_or_dir, f"rank{rank_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        print(f"% Using {path}")
        with open(path) as f:
            return json.load(f)

    # Aggregation across ranks
    rank_datas = load_ranks(path_or_dir, exclude_ranks)

    if rank == "fastest":
        return find_fastest_rank(rank_datas, metric, minival_metric,
                                 target, tolerance)

    return aggregate_data(rank_datas, metric, rank, minival_metric)


# ---------------------------------------------------------------------------
# Core TTA logic
# ---------------------------------------------------------------------------

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
    # return np.array(mins), np.round(np.array(vals), 2)


def find_crossing(m, v, target, hib=False, tolerance=0.0):
    """Find time where metric crosses target via linear interpolation."""
    eff = (target - tolerance) if hib else (target + tolerance)
    for i in range(1, len(v)):
        if hib:
            if v[i] >= eff and v[i - 1] < eff:
                frac = (eff - v[i - 1]) / (v[i] - v[i - 1])
                return m[i - 1] + frac * (m[i] - m[i - 1])
        else:
            if v[i] <= eff and v[i - 1] > eff:
                frac = (v[i - 1] - eff) / (v[i - 1] - v[i])
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tta(input_a, input_b, metric, labels=("BSP", "SA-INA"),
             minival_metric=None, target=None, output=None, smooth=False,
             title=None, epochs=None, tolerance=0.0,
             rank="rank0", exclude_ranks=None):
    import plot_conf
    import matplotlib.pyplot as plt

    hib = higher_is_better(metric)

    da = load_input(input_a, rank, exclude_ranks, metric, minival_metric,
                    target, tolerance)
    db = load_input(input_b, rank, exclude_ranks, metric, minival_metric,
                    target, tolerance)

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
        crossings.append(find_crossing(m, v, target, hib, tolerance) if target else None)

    if target and tolerance > 0:
        eff = (target - tolerance) if hib else (target + tolerance)
        print(f"% Target: {target}, tolerance: ±{tolerance}, effective: {eff}")
    print(f"% Rank aggregation: {rank}")
    if exclude_ranks:
        print(f"% Excluded ranks: {exclude_ranks}")
    print_tex_coordinates(series, crossings, labels, target)

    # Plot
    fig, ax = plt.subplots(figsize=(plot_conf.COLUMN_WIDTH, 2.2))
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
        valid_crossings = [tc for tc in crossings if tc is not None]
        ax.axhline(target, color="#4B5157", ls="--", lw=0.7, zorder=2)
        eff_target = (target - tolerance) if hib else (target + tolerance)
        if tolerance > 0:
            ax.axhspan(min(target, eff_target), max(target, eff_target),
                       color="#4B5157", alpha=0.07, zorder=1)

        if len(valid_crossings) == 2:
            tc_min = min(valid_crossings)
            tc_max = max(valid_crossings)
            ylim = ax.get_ylim()
            if hib:
                ymax_frac = (ylim[1] - target) / (ylim[1] - ylim[0])
                ax.axvspan(tc_min, tc_max, ymin=1 - ymax_frac, ymax=1,
                           color="green", alpha=0.06, zorder=1)
            else:
                ymax_frac = (target - ylim[0]) / (ylim[1] - ylim[0])
                ax.axvspan(tc_min, tc_max, ymin=0, ymax=ymax_frac,
                           color="green", alpha=0.06, zorder=1)

            for i, tc in enumerate(crossings):
                if tc:
                    ax.axvline(tc, color=colors[i], ls=":", lw=0.6, alpha=0.5,
                               ymax=0.95, zorder=2)
            sp = tc_max / tc_min
            arrow_y = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.35
            ax.annotate("", xy=(tc_min, arrow_y), xytext=(tc_max, arrow_y),
                         arrowprops=dict(arrowstyle="<->", color="black", lw=0.6, alpha=0.4))
            ax.text((tc_min + tc_max) / 2, arrow_y, f"{sp:.2f}x",
                    ha="center", va="center", fontsize=6,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="gray",
                              lw=0.3, alpha=0.9),
                    color="0.4", fontweight="bold")
        elif len(valid_crossings) == 1:
            tc = valid_crossings[0]
            idx = crossings.index(tc)
            ax.axvline(tc, color=colors[idx], ls=":", lw=0.6, alpha=0.5,
                       ymax=0.95, zorder=2)
            other = 1 - idx
            direction = "above" if hib else "below"
            print(f"% WARNING: only {labels[idx]} crosses target; "
                  f"{labels[other]} never reaches {direction} {target}")

    # Fixed-epoch comparison
    end_times = [series[i][0][-1] for i in range(2)]
    end_vals = [series[i][1][-1] for i in range(2)]
    ylim = ax.get_ylim()
    for i in range(2):
        ax.axvline(end_times[i], color=colors[i], ls=(0, (5, 3)), lw=0.7,
                   alpha=0.5, zorder=2)
    arrow_y = ylim[1] - (ylim[1] - ylim[0]) * 0.15
    ratio = max(end_times) / min(end_times)
    ax.annotate("", xy=(min(end_times), arrow_y), xytext=(max(end_times), arrow_y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=0.6, alpha=0.4))
    ax.text((end_times[0] + end_times[1]) / 2, arrow_y, f"{ratio:.2f}x",
            ha="center", va="center", fontsize=6,
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="gray",
                      lw=0.3, alpha=0.9),
            color="0.4", fontweight="bold")

    print(f"\n% Fixed-epoch ({epochs or 'auto'}):")
    print(f"%   {labels[0]} ends at {end_times[0]:.2f} min, {metric}={end_vals[0]:.2f}")
    print(f"%   {labels[1]} ends at {end_times[1]:.2f} min, {metric}={end_vals[1]:.2f}")
    print(f"%   Time ratio: {ratio:.2f}x ({abs(end_times[0] - end_times[1]):.2f} min saved)")

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(metric_label(metric))
    if title:
        ax.set_title(title, fontweight="bold", fontfamily="sans-serif")
    ax.legend(loc="lower right" if hib else "upper right")
    fig.tight_layout()

    if output:
        fig.savefig(output)
        print(f"\nSaved {output}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Time-to-accuracy plot with multi-rank aggregation")
    p.add_argument("--input_a", default=None, help="Baseline: directory or rank*.json file")
    p.add_argument("--input_b", default=None, help="Optimized: directory or rank*.json file")
    p.add_argument("--metric", default="val_ppl")
    p.add_argument("--minival", default=None)
    p.add_argument("--labels", nargs=2, default=["SU", "SA"])
    p.add_argument("--target", type=float, default=None)
    p.add_argument("--tolerance", type=float, default=0.0, help="Buffer zone around target")
    p.add_argument("--rank", type=str, default="rank0", help="0, 1, ..., avg, best, fastest (default: rank0)")
    p.add_argument("--exclude-ranks", type=int, nargs="*", default=None, help="Rank IDs to exclude (e.g. straggler)")
    p.add_argument("--smooth", action="store_true")
    p.add_argument("--title", default=None)
    p.add_argument("--epochs", type=int, default=None, help="Max epochs to show (auto-matches if unset)")
    p.add_argument("-o", "--output", default=None)
    a = p.parse_args()

    from pathlib import Path

    DIR = Path(__file__).parent

    # # resnet
    # a.input_a = DIR / "resnet/aggressive/su/"
    # a.input_b = DIR / "resnet/aggressive/sa"
    # a.metric = "val_top5"
    # a.target = 90
    # a.tolerance = 0.01
    # a.rank = "avg"
    # a.exclude_ranks = [1]
    # a.smooth = True

    # gpt2
    # a.input_a = DIR / "gpt2/aggressive/su"
    # a.input_b = DIR / "gpt2/aggressive/sa"
    # a.metric = "val_ppl"
    # a.target = 28.00
    # a.tolerance = 0.01
    # a.rank = "avg"
    # a.minival = "mini_val_ppl"
    # a.exclude_ranks = [1]
    # a.smooth = True

    # roberta
    a.input_a = DIR / "roberta/aggressive/su"
    a.input_b = DIR / "roberta/aggressive/sa"
    a.metric = "val_f1"
    a.target = 80
    a.rank = "avg"
    a.minival = "mini_val_f1"
    a.exclude_ranks = [1]
    a.smooth = True
    a.tolerance = 0.01
    a.epochs = 4

    plot_tta(a.input_a, a.input_b, a.metric, a.labels, a.minival, a.target,
             a.output, a.smooth, a.title, a.epochs, a.tolerance,
             a.rank, a.exclude_ranks)
    

