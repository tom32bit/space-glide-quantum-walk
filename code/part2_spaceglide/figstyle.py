"""Shared figure style for all paper figures.

Categorical palette (fixed order, validated for colour-vision-deficiency
separation on a light surface; the two lightest hues have low contrast against
white, so every use of them is accompanied by a direct label or a legend):
    C1 blue, C2 orange, C3 aqua, C4 yellow.
Sequential data (gap magnitudes, weights) use ONE hue, light -> dark (SEQ).
Text and axes use ink colours, never a series colour.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, RULE = "#0b0b0b", "#52514e", "#c9c8c2"
NEUTRAL = "#f0efec"

SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
BINARY = ListedColormap([NEUTRAL, C1])                     # index 0 / 1

# the four glide classes (zeta_0, zeta_pi), fixed colour per class
CLASS_ORDER = [(0, 0), (0, 1), (1, 0), (1, 1)]
CLASS_COLOR = {(0, 0): C1, (0, 1): C2, (1, 0): C3, (1, 1): C4}
CLASS_CMAP = ListedColormap([CLASS_COLOR[c] for c in CLASS_ORDER])

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.7,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "text.color": INK,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "mathtext.fontset": "cm",
})


def class_index(z0, zp):
    """Integer 0..3 for the colour map CLASS_CMAP."""
    return 2 * z0 + zp
