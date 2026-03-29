#!/usr/bin/env python3
"""
Time-to-accuracy plot with multi-rank aggregation.

Usage:
    python plot_tta.py --input-a data/su/ --input-b data/sa/ \\
        --metric val_ppl --target 28 --rank avg --exclude-ranks-b 3 -o plot.pdf

Rank options: rank0, rank1, ..., avg, best, fastest
"""
import json, argparse, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

METRIC_LABELS = {
    "val_ppl": "Validation Perplexity",     "train_ppl": "Training Perplexity",
    "val_loss": "Validation Loss",           "train_loss": "Training Loss",
    "mini_val_ppl": "Validation Perplexity", "mini_val_loss": "Validation Loss",
    "val_acc": "Validation Accuracy (\\%)", "train_acc": "Training Accuracy (\\%)",
    "val_top1": "Top-1 Accuracy (\\%)",     "val_top5": "Top-5 Accuracy (\\%)",
    "f1": "F1 Score", "precision": "Precision", "recall": "Recall",
    "exact_match": "Exact Match",
}
LOWER_IS_BETTER = {"loss", "ppl"}

def hib(metric):  # higher-is-better
    return not any(k in metric for k in LOWER_IS_BETTER)

def metric_label(m):
    return METRIC_LABELS.get(m, m.replace("_", r"\_"))


# ---------------------------------------------------------------------------
# Loading & aggregation
# ---------------------------------------------------------------------------

def load_ranks(directory, exclude=None):
    exclude = set(exclude or [])
    pairs = [(int(os.path.basename(p).replace("rank","").replace(".json","")), p)
             for p in sorted(glob.glob(os.path.join(directory, "rank*.json")))]
    pairs = [(rid, p) for rid, p in pairs if rid not in exclude]
    if not pairs:
        raise FileNotFoundError(f"No rank*.json in {directory} (excluded: {exclude})")
    print(f"% Ranks {[r for r,_ in pairs]} from {directory}")
    return {rid: json.load(open(p)) for rid, p in pairs}


def aggregate_ranks(rank_datas, metric, mode, minival_metric=None):
    ids = sorted(rank_datas.keys())
    ref = rank_datas[ids[0]]
    epoch_keys = sorted(ref["epochs"], key=int)[:min(len(d["epochs"]) for d in rank_datas.values())]
    pick = (lambda vs: max(vs) if hib(metric) else min(vs)) if mode == "best" else np.mean

    agg_epochs, agg_minival = {}, {}
    for ek in epoch_keys:
        ref_ep = ref["epochs"][ek]
        agg_epochs[ek] = {**ref_ep, metric: float(pick([rank_datas[r]["epochs"][ek][metric] for r in ids]))}

        if minival_metric:
            all_steps = set().union(*[rank_datas[r].get("minival",{}).get(ek,{}).keys() for r in ids])
            mv_pick = (lambda vs: max(vs) if hib(minival_metric) else min(vs)) if mode == "best" else np.mean
            agg_minival[ek] = {
                sk: {minival_metric: float(mv_pick([
                    rank_datas[r]["minival"][ek][sk][minival_metric]
                    for r in ids if sk in rank_datas[r].get("minival",{}).get(ek,{})]))}
                for sk in sorted(all_steps, key=int)
                if any(sk in rank_datas[r].get("minival",{}).get(ek,{}) for r in ids)
            }

    return {"epochs": agg_epochs, **({"minival": agg_minival} if agg_minival else {})}


def load_input(path, rank="rank0", exclude=None, metric=None,
               minival_metric=None, target=None, tolerance=0.0, exclude_pre_train=False):
    path = str(path)
    if os.path.isfile(path):
        return json.load(open(path))
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Not a file or directory: {path}")
    if rank.isdigit(): rank = f"rank{rank}"
    if rank.startswith("rank"):
        return json.load(open(os.path.join(path, f"{rank}.json")))

    rank_datas = load_ranks(path, exclude)

    if rank == "fastest":
        best_time, best_rid = None, None
        for rid, data in rank_datas.items():
            m, v = extract_curve(data, metric, minival_metric, exclude_pre_train=exclude_pre_train)
            tc = find_crossing(m, v, target, hib(metric), tolerance)
            if tc and (best_time is None or tc < best_time):
                best_time, best_rid = tc, rid
        if best_rid is None:
            finals = {r: list(d["epochs"].values())[-1][metric] for r, d in rank_datas.items()}
            best_rid = (max if hib(metric) else min)(finals, key=finals.get)
            print(f"% fastest: no crossing, using rank {best_rid} (final={finals[best_rid]:.2f})")
        else:
            print(f"% fastest: rank {best_rid} at {best_time:.2f} min")
        return rank_datas[best_rid]

    if rank in ("avg", "best"):
        return aggregate_ranks(rank_datas, metric, rank, minival_metric)

    raise ValueError(f"Unknown rank: {rank!r}")


# ---------------------------------------------------------------------------
# Curve & crossing
# ---------------------------------------------------------------------------

def extract_curve(data, metric, minival_metric=None, max_epochs=None, exclude_pre_train=False):
    epochs = data["epochs"]
    minival = data.get("minival", {})
    cum, mins, vals = 0.0, [], []
    for ek in sorted(epochs, key=int)[:max_epochs]:
        ep = epochs[ek]
        dur, steps = ep["epoch_time"], ep["steps"]
        for sk in sorted(minival.get(ek, {}), key=int):
            if not (exclude_pre_train and int(sk) == 0):
                mins.append((cum + int(sk) / steps * dur) / 60)
                vals.append(minival[ek][sk][minival_metric])
        cum += dur
        mins.append(cum / 60)
        vals.append(ep[metric])
    return np.array(mins), np.array(vals)


def find_crossing(m, v, target, higher=False, tolerance=0.0):
    eff = (target - tolerance) if higher else (target + tolerance)
    for i in range(1, len(v)):
        if (higher and v[i] >= eff > v[i-1]) or (not higher and v[i] <= eff < v[i-1]):
            frac = abs(eff - v[i-1]) / abs(v[i] - v[i-1])
            return m[i-1] + frac * (m[i] - m[i-1])
    return None


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def annotate_span(ax, x0, x1, y, label, color0, color1, colors):
    """Draw a <-> arrow with a speedup label between two x positions."""
    ax.annotate("", xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="<->", color="black", lw=0.6, alpha=0.4))
    ax.text((x0 + x1) / 2, y, label, ha="center", va="center", fontsize=6,
            color="0.4", fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="gray", lw=0.3, alpha=0.9))


def plot_tta(input_a, input_b, metric, labels=("SU", "SA"),
             minival_metric=None, target=None, output=None, smooth=False,
             title=None, epochs=None, tolerance=0.0, rank="rank0",
             exclude_ranks_a=None, exclude_ranks_b=None, exclude_pre_train=False):
    import plot_conf, matplotlib.pyplot as plt

    higher = hib(metric)
    kw = dict(metric=metric, minival_metric=minival_metric,
              target=target, tolerance=tolerance, exclude_pre_train=exclude_pre_train)
    da = load_input(input_a, rank, exclude_ranks_a, **kw)
    db = load_input(input_b, rank, exclude_ranks_b, **kw)

    if epochs is None and len(da["epochs"]) != len(db["epochs"]):
        epochs = min(len(da["epochs"]), len(db["epochs"]))

    series, crossings = [], []
    for data, label in zip([da, db], labels):
        m, v = extract_curve(data, metric, minival_metric, epochs, exclude_pre_train)
        series.append((m, v, label))
        crossings.append(find_crossing(m, v, target, higher, tolerance) if target else None)

    # Log
    for (m, v, label), tc in zip(series, crossings):
        print(f"% {label}: " + " ".join(f"({mi:.2f},{vi:.2f})" for mi, vi in zip(m, v)))
        if tc: print(f"% {label} crosses {target} at {tc:.2f} min")
    if all(crossings): print(f"% Speedup: {max(crossings)/min(crossings):.2f}x")

    colors = ["#E63946", "#2A9D8F"]
    fig, ax = plt.subplots(figsize=(plot_conf.COLUMN_WIDTH, 2.2))

    for i, (m, v, label) in enumerate(series):
        kws = dict(color=colors[i], lw=1.2, label=label, zorder=3)
        if smooth and not minival_metric:
            from scipy.interpolate import PchipInterpolator
            ms = np.linspace(m[0], m[-1], 300)
            ax.plot(ms, PchipInterpolator(m, v)(ms), **kws)
            ax.scatter(m, v, color=colors[i], s=14, marker="so"[i],
                       edgecolors="white", linewidths=0.4, zorder=4)
        else:
            ax.plot(m, v, marker="so"[i], ms=3.5, markeredgecolor="white",
                    markeredgewidth=0.4, **kws)

    ylim = ax.get_ylim()
    yspan = ylim[1] - ylim[0]

    if target:
        ax.axhline(target, color="#4B5157", ls="--", lw=0.7, zorder=2)
        if tolerance > 0:
            eff = (target - tolerance) if higher else (target + tolerance)
            ax.axhspan(min(target, eff), max(target, eff), color="#4B5157", alpha=0.07)
        valid = [(i, tc) for i, tc in enumerate(crossings) if tc]
        if len(valid) == 2:
            tcs = [tc for _, tc in valid]
            frac = ((ylim[1] - target) if higher else (target - ylim[0])) / yspan
            ax.axvspan(min(tcs), max(tcs), ymin=1-frac if higher else 0,
                       ymax=1 if higher else frac, color="green", alpha=0.06)
            for i, tc in enumerate(crossings):
                ax.axvline(tc, color=colors[i], ls=":", lw=0.6, alpha=0.5, ymax=0.95)
            annotate_span(ax, min(tcs), max(tcs), ylim[0] + yspan*0.35,
                          f"{max(tcs)/min(tcs):.2f}x", *colors, colors)
        elif len(valid) == 1:
            i, tc = valid[0]
            ax.axvline(tc, color=colors[i], ls=":", lw=0.6, alpha=0.5, ymax=0.95)
            print(f"% WARNING: only {labels[i]} crosses target")

    # Fixed-epoch wall-clock ratio
    end_t = [s[0][-1] for s in series]
    for i in range(2):
        ax.axvline(end_t[i], color=colors[i], ls=(0,(5,3)), lw=0.7, alpha=0.5)
    annotate_span(ax, min(end_t), max(end_t), ylim[1] - yspan*0.15,
                  f"{max(end_t)/min(end_t):.2f}x", *colors, colors)
    print(f"% Fixed-epoch: {labels[0]}={end_t[0]:.2f}min, {labels[1]}={end_t[1]:.2f}min, "
          f"ratio={max(end_t)/min(end_t):.2f}x")

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(metric_label(metric))
    if title: ax.set_title(title, fontweight="bold", fontfamily="sans-serif")
    ax.legend(loc="lower right" if higher else "upper right")
    fig.tight_layout()
    if output:
        fig.savefig(output)
        print(f"Saved {output}")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Time-to-accuracy plot")
    p.add_argument("--input-a")
    p.add_argument("--input-b")
    p.add_argument("--metric",            default="val_ppl")
    p.add_argument("--minival",           default=None)
    p.add_argument("--labels",            nargs=2, default=["SU", "SA"])
    p.add_argument("--target",            type=float, default=None)
    p.add_argument("--tolerance",         type=float, default=0.0)
    p.add_argument("--rank",              default="rank0", help="rank0..N | avg | best | fastest")
    p.add_argument("--exclude-ranks-a",   type=int, nargs="*", default=None)
    p.add_argument("--exclude-ranks-b",   type=int, nargs="*", default=None)
    p.add_argument("--epochs",            type=int, default=None)
    p.add_argument("--smooth",            action="store_true")
    p.add_argument("--exclude-pre-train", action="store_true")
    p.add_argument("--title",             default=None)
    p.add_argument("-o", "--output",      default=None)
    a = p.parse_args()

    from pathlib import Path
    DIR = Path(__file__).parent

    # ---- Uncomment one block to run directly ----

    # # resnet
    # a.input_a = DIR / "resnet/aggressive/su/"
    # a.input_b = DIR / "resnet/aggressive/sa"
    # a.metric = "val_top5"
    # a.target = 90
    # a.tolerance = 0.01
    # a.rank = "avg"
    # a.exclude_ranks_a = None
    # a.exclude_ranks_b = [1]
    # a.smooth = True

    # gpt2
    a.input_a = DIR / "gpt2/moderate/su"
    a.input_b = DIR / "gpt2/moderate/sa"
    a.metric = "val_ppl"
    a.target = 28.00
    a.tolerance = 0.01
    a.rank = "best"
    a.minival = "mini_val_ppl"
    a.exclude_ranks_a = [1]
    a.exclude_ranks_b = [1]
    a.smooth = True
    a.epochs = 6

    # roberta
    # a.input_a = DIR / "roberta/moderate/su"
    # a.input_b = DIR / "roberta/moderate/sa"
    # a.exclude_ranks_a = [1]
    # a.exclude_ranks_b = [1]
    # a.metric = "val_f1"
    # a.target = 80
    # a.rank = "avg"
    # a.minival = "mini_val_f1"
    # a.smooth = True
    # a.tolerance = 0.01
    # a.epochs = 4

    #qwen
    # a.input_a = DIR / "qwen25-metamath40k/aggressive-75/su"
    # a.input_b = DIR / "qwen25-metamath40k/aggressive-75/sa"
    # a.exclude_pre_train = 0
    # a.metric = "val_ppl"
    # a.target = 1.30
    # a.rank = "avg"
    # a.minival = "mini_val_ppl"
    # a.exclude_ranks_a = None
    # a.exclude_ranks_b = [1]
    # a.smooth = False
    # # a.tolerance = 0.01
    # a.epochs = 3

    plot_tta(a.input_a, a.input_b, a.metric, a.labels, a.minival, a.target,
             a.output, a.smooth, a.title, a.epochs, a.tolerance, a.rank,
             exclude_ranks_a=a.exclude_ranks_a, exclude_ranks_b=a.exclude_ranks_b,
             exclude_pre_train=a.exclude_pre_train)