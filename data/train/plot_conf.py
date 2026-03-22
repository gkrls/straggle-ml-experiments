import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler

# IEEE double-column layout
COLUMN_WIDTH = 3.5    # single column
FULL_WIDTH = 7.16     # full width
# For N subfigures across full width:
SUB3_WIDTH = FULL_WIDTH / 3  # ~2.39in
SUB4_WIDTH = FULL_WIDTH / 4  # ~1.79in

FONT_SIZE = 7

COLORS = {
    "blue":   "#2563eb",
    "red":    "#dc2626",
    "green":  "#16a34a",
    "orange": "#ea580c",
    "purple": "#7c3aed",
    "gray":   "#6b7280",
}
COLOR_LIST = list(COLORS.values())

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",

    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,

    "figure.dpi": 300,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "savefig.facecolor": "white",

    # Thicker lines for small plots
    "lines.linewidth": 1.3,
    "lines.markersize": 3,

    # Axes
    "axes.linewidth": 0.5,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#333333",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.prop_cycle": cycler("color", COLOR_LIST),
    "axes.labelpad": 2,
    "axes.titlepad": 4,

    # Grid
    "grid.color": "#e5e7eb",
    "grid.linewidth": 0.4,
    "grid.alpha": 1.0,

    # Ticks - compact
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.direction": "out",
    "ytick.direction": "out",

    # Legend - tight
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#e5e7eb",
    "legend.borderpad": 0.2,
    "legend.handlelength": 1.0,
    "legend.handletextpad": 0.3,
    "legend.labelspacing": 0.2,
    "legend.columnspacing": 0.8,
    "legend.fancybox": False,

    # Tight subplots
    "figure.subplot.left": 0.0,
    "figure.subplot.right": 1.0,
    "figure.subplot.bottom": 0.0,
    "figure.subplot.top": 1.0,
    "figure.subplot.wspace": 0.05,
    "figure.subplot.hspace": 0.05,

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})