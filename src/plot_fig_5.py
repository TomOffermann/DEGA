import matplotlib as mpl

# ─────── embed only Type-1/TrueType & disable LaTeX ───────
mpl.rcParams["pdf.fonttype"] = 42  # embed TrueType in PDF
mpl.rcParams["ps.fonttype"] = 42  # embed TrueType in PS
mpl.rcParams["text.usetex"] = False

# pick a font that actually contains “①”, “②”, …
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]

# use the same font for math
mpl.rcParams["mathtext.fontset"] = "dejavusans"

import matplotlib.pyplot as plt

from visualize import Plotter
from matplotlib.ticker import NullFormatter
from math import log, floor, ceil
import numpy as np

# -------------------------------- prepare figure & Plotter -------------
fig, (ax_lo, ax_om, ax_lfhw) = plt.subplots(
    1, 3, figsize=(12, 4), sharex=False, sharey=False
)
p = Plotter(data_dir="data", run_name="COMPARISON_DIFFERENT_DEGA")

# -------------------------------- series definitions -------------------
series = [
    #  LO  (axis 0, ÷ n²)
    dict(
        algorithm="DEGA",
        benchmark="LO",
        param_desc="lamb=n^(2/3)",
        label=r"$DEGA^{\lambda=n^{2/3}}$",
        axis=0,
        color="#D443FF",
        marker="o",
    ),
    dict(
        algorithm="DEGA_A",
        benchmark="LO",
        param_desc="",
        label=r"$DEGA_+$",
        axis=0,
        color="#7143FF",
        marker="s",
    ),
    dict(
        algorithm="DEGA_B",
        benchmark="LO",
        param_desc="",
        label=r"$DEGA_{LO}$",
        axis=0,
        color="#43CCFF",
        marker="^",
    ),
    #  OM  (axis 1, ÷ n ln n)
    dict(
        algorithm="DEGA",
        benchmark="OM",
        param_desc="lamb=n^(2/3)",
        label=r"$DEGA_{Proof}^{\lambda=n^{2/3}}$",
        axis=1,
        color="#D443FF",
        marker="o",
    ),
    dict(
        algorithm="DEGA_A",
        benchmark="OM",
        param_desc="",
        label=r"$DEGA_+$",
        axis=1,
        color="#7143FF",
        marker="s",
    ),
    dict(
        algorithm="DEGA_B",
        benchmark="OM",
        param_desc="",
        label=r"$DEGA_{LO}$",
        axis=1,
        color="#43CCFF",
        marker="^",
    ),
    #  LFHW  (axis 2, raw)
    dict(
        algorithm="DEGA",
        benchmark="LFHW",
        param_desc="lamb=n^(2/3)",
        label=r"$DEGA^{\lambda=n^{2/3}}$",
        axis=2,
        color="#D443FF",
        marker="o",
    ),
    dict(
        algorithm="DEGA_A",
        benchmark="LFHW",
        param_desc="",
        label=r"$DEGA_+$",
        axis=2,
        color="#7143FF",
        marker="s",
    ),
    dict(
        algorithm="DEGA_B",
        benchmark="LFHW",
        param_desc="",
        label=r"$DEGA_{LO}$",
        axis=2,
        color="#43CCFF",
        marker="^",
    ),
]

# -------------------------------- axis-wise normalisation --------------
axis_norms = {0: "n2", 1: "nlogn", 2: "nlogn"}

# -------------------------------- draw ---------------------------------
p.plot_any(
    series,
    axes=[ax_lo, ax_om, ax_lfhw],
    loglog=True,
    plot_std=True,
    ylabel=[
        r"$\bar T(n)$ / $n^{2}$",
        r"$\bar T(n)$ / $n\ \ln n$",
        r"$\bar T(n)$ / $n\ \ln n$",
    ],
    legend_axes=[0],  # one data-legend is enough
    ref_funcs=None,
    ref_legend=False,
    title=[r"$LO$", r"$OM$", r"$LFHW$"],
    axis_norms=axis_norms,
)

# Per-panel n-grids (for diagonal labels only at these ticks).
n_lo = [100, 129, 166, 215, 278, 359, 464, 599, 774, 1000]
n_om = [100, 166, 278, 464, 774, 1291, 2154, 3593, 5994, 10000]
n_lfhw = [100, 154, 238, 368, 568, 878, 1357, 2096, 3237, 4999]

for ax, ticks in ((ax_lo, n_lo), (ax_om, n_om), (ax_lfhw, n_lfhw)):
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(v) for v in ticks], rotation=45)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", labelbottom=False)

# Keep a small log-space gap before first and after last n (as in reference plot).
pad = 1.15
ax_lo.set_xlim(n_lo[0] / pad, n_lo[-1] * pad)
ax_om.set_xlim(n_om[0] / pad, n_om[-1] * pad)
ax_lfhw.set_xlim(n_lfhw[0] / pad, n_lfhw[-1] * pad)

# Explicit per-axis y autoscaling (independent axes, with small log-space padding).
for ax in (ax_lo, ax_om, ax_lfhw):
    vals = []
    for line in ax.get_lines():
        y = np.asarray(line.get_ydata(), dtype=float)
        y = y[np.isfinite(y) & (y > 0)]
        if y.size:
            vals.append(y)
    if vals:
        y_all = np.concatenate(vals)
        ax.set_ylim(y_all.min() / 1.15, y_all.max() * 1.15)

# LO-only: denser y tick labels (1-2-5 ticks per decade), as requested.
y_min, y_max = ax_lo.get_ylim()
lo = floor(log(y_min, 10))
hi = ceil(log(y_max, 10))
y_ticks_lo = []
for e in range(lo, hi + 1):
    for m in (1, 2, 5):
        y = m * (10**e)
        if y_min <= y <= y_max:
            y_ticks_lo.append(y)
ax_lo.set_yticks(y_ticks_lo)
ax_lo.set_yticklabels([f"{y:g}" for y in y_ticks_lo])

# OM/LFHW: add a few horizontal gray guide lines (in-panel values only).
guides = {
    ax_om: [2, 3, 4, 6, 10],
    ax_lfhw: [3, 4, 6, 10],
}
# Ensure runtime curves stay above any scaffold lines (same visual layering as LO).
for ax in (ax_om, ax_lfhw):
    for ln in ax.lines:
        ln.set_zorder(3)

# Reuse the exact LO horizontal grid style for consistency.
lo_y_grid = [ln for ln in ax_lo.get_ygridlines() if ln.get_visible()]
if lo_y_grid:
    lo_grid_style = dict(
        color=lo_y_grid[0].get_color(),
        linestyle=lo_y_grid[0].get_linestyle(),
        linewidth=lo_y_grid[0].get_linewidth(),
        alpha=lo_y_grid[0].get_alpha(),
    )
else:
    lo_grid_style = dict(color="#b0b0b0", linestyle=":", linewidth=0.8, alpha=0.5)

for ax, ys in guides.items():
    y0, y1 = ax.get_ylim()
    for y in ys:
        if y0 <= y <= y1:
            ax.axhline(y, zorder=0, **lo_grid_style)

# Add more horizontal spacing so y-labels do not intrude into neighboring panels.
fig.subplots_adjust(wspace=0.32)

# save (optional)
import os

out = "./plots/COMPARISON_DIFFERENT_DEGA.pdf"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, format="pdf", dpi=300, transparent=True)
plt.show()
