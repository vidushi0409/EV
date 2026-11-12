"""Shared chart colours so the notebook, report and app look consistent."""
from matplotlib.colors import LinearSegmentedColormap

# Fixed categorical order (one colour per model, never reassigned)
MODEL_COLORS = {
    "Linear Regression": "#2a78d6",   # blue
    "Random Forest": "#eb6834",       # orange
    "LightGBM": "#1baf7a",            # aqua
    "Naive (lag-1)": "#8a8a85",       # neutral grey - reference only
    "Naive (lag-168)": "#b5b4ae",
}
ACTUAL_COLOR = "#2b2b29"   # dark ink for observed values
INK = "#2b2b29"
MUTED = "#6b6b66"
GRID = "#e6e5e1"

# Diverging blue <-> red with a neutral grey midpoint (for correlations, -1..1)
DIVERGING = LinearSegmentedColormap.from_list(
    "ev_diverging", ["#2a78d6", "#f0efec", "#e34948"])


def apply_matplotlib_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "axes.titleweight": "bold",
    })
