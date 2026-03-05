# plot_any version of “LO-different-lambda-7500”

import matplotlib as mpl

# ─────── embed only Type-1/TrueType & disable LaTeX ───────
mpl.rcParams['pdf.fonttype']    = 42   # embed TrueType in PDF
mpl.rcParams['ps.fonttype']     = 42   # embed TrueType in PS
mpl.rcParams['text.usetex']     = False

# pick a font that actually contains “①”, “②”, …  
mpl.rcParams['font.family']     = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']

# use the same font for math  
mpl.rcParams['mathtext.fontset'] = 'dejavusans'

from visualize import Plotter
import matplotlib.pyplot as plt
from math import log, floor, ceil
import os

p = Plotter(data_dir="data", run_name="LO_ASYMPTOTIC_DEGA")

fig, (ax_opt, ax_diff, ax_misc) = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

# ── which λ goes to which subplot ─────────────────────────────────────
grouping = {
    "lamb=(n*log(n))**(2/3)": 0,
    "lamb=n^(2/3)": 0,
    "lamb=sqrt(n)": 1,
    "lamb=2": 1,
    "lamb=sqrt(n*log(n))": 2,
    "lamb=n^(1/3)": 2,
}

label_of = {
    "lamb=(n*log(n))**(2/3)": r"$\lambda_1=(n\ln n)^{2/3}$",
    "lamb=n^(2/3)": r"$\lambda_2=n^{2/3}$",
    "lamb=2": r"$\lambda_3=2$",
    "lamb=sqrt(n)": r"$\lambda_4=\sqrt{n}$",
    "lamb=n^(1/3)": r"$\lambda_5=n^{1/3}$",
    "lamb=sqrt(n*log(n))": r"$\lambda_6=\sqrt{n\ln n}$",
}

# explicit colours to keep the old order (yellow skipped automatically)
colour_of = {
    "lamb=(n*log(n))**(2/3)": "#FF9643",
    "lamb=n^(2/3)": "#FF435D",
    "lamb=2": "#D443FF",
    "lamb=sqrt(n)": "#7143FF",
    "lamb=n^(1/3)": "#43CCFF",
    "lamb=sqrt(n*log(n))": "#43FF76",
}
markers = ["o", "s", "^", "d", "v", "P"]

# ── build the series-list for plot_any() ───────────────────────────────
series = []
for idx, raw in enumerate(colour_of):  # preserves colour order
    axes_target = grouping[raw]
    if not isinstance(axes_target, (list, tuple)):
        axes_target = [axes_target]
    series.append(
        dict(
            algorithm="DEGA",
            benchmark="LO",
            param_desc=raw,
            label=label_of[raw],
            axis=axes_target,
            color=colour_of[raw],
            marker=markers[idx % len(markers)],
        )
    )

# ── reference curves (unchanged) ───────────────────────────────────────
refs = [
    dict(
        func=lambda n: 1.2 * n ** (5 / 3) * log(n) ** (2 / 3),
        label=r"$n^{5/3}\ln^{2/3} n$",
        color="#FF9643",
        linestyle=":",
        axis=[0],
    ),
    dict(
        func=lambda n: 0.45 * n ** (5 / 3) * log(n),
        label=r"$n^{5/3}\ln n$",
        color="#FF435D",
        linestyle=":",
        axis=[0],
    ),
    dict(
        func=lambda n: 0.17 * n**2 * log(n),
        label=r"$n^{2}\ln n$",
        color="#D443FF",
        linestyle=":",
        axis=[1],
    ),
    dict(
        func=lambda n: 0.3 * n ** (7 / 4) * log(n),
        label=r"$n^{7/4}\ln n$",
        color="#7143FF",
        linestyle=":",
        axis=[1],
    ),
    dict(
        func=lambda n: 0.25 * n ** (11 / 6) * log(n),
        label=r"$n^{11/6}\ln n$",
        color="#43CCFF",
        linestyle=":",
        axis=[2],
    ),
    dict(
        func=lambda n: 0.45 * n ** (7 / 4) * log(n) ** (3 / 4),
        label=r"$n^{7/4}\ln^{3/4} n$",
        color="#43FF76",
        linestyle=":",
        axis=[2],
    ),
    dict(
        func=lambda n: n**2,
        label=r"$n^{2}$",
        color="gray",
        linestyle="--",
        axis=[0, 1, 2],
    ),
    dict(
        func=lambda n: 1.5 * n ** (5 / 3),
        label=r"$n^{5/3}$",
        color="gray",
        linestyle="-.",
        axis=[0, 1, 2],
    ),
]

# ── per-axis normalisation (first two panels ÷ n²) ────────────────────
axis_norms = {0: "n2", 1: "n2", 2: "n2"}

# ── draw everything with the new API ───────────────────────────────────
p.plot_any(
    series,
    axes=[ax_opt, ax_diff, ax_misc],
    loglog=True,
    plot_std=True,
    ylabel=[r"$\bar T(n)/n^2$", r"$\bar T(n)/n^2$", r"$\bar T(n)/n^2$"],
    title="",
    xlim=(50, 10000),
    ref_funcs=refs,
    ref_legend="below",  # horizontal legend for references
    axis_norms=axis_norms,
    legend_axes=[0, 1, 2],  # data legend on every panel
)

# Show more y-tick labels on log scale (reviewer request): 1-2-5 ticks per decade.
for ax in (ax_opt, ax_diff, ax_misc):
    y_min, y_max = ax.get_ylim()
    lo = floor(log(y_min, 10))
    hi = ceil(log(y_max, 10))
    yticks = []
    for e in range(lo, hi + 1):
        for m in (1, 2, 5):
            y = m * (10 ** e)
            if y_min <= y <= y_max:
                yticks.append(y)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:g}" for y in yticks])

# save (optional)
out = "./plots/LO_ASYMPTOTIC_DEGA.pdf"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, format="pdf", dpi=300, transparent=True)
# plt.show()
