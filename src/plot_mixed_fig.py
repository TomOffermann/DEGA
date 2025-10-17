#!/usr/bin/env python3

import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42  # embed TrueType in PDF
mpl.rcParams["ps.fonttype"] = 42  # embed TrueType in PS
mpl.rcParams["text.usetex"] = False

# pick a font that actually contains “①”, “②”, …
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]

# use the same font for math
mpl.rcParams["mathtext.fontset"] = "dejavusans"
 
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Sequence
from collections import defaultdict
import numpy as np
import json, re
from math import log
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# -----------------------------------------------------------------------------
# Paste in your Plotter class here (the full plotter.py), or import it if you
# already have it on PYTHONPATH:
from visualize import Plotter

# -----------------------------------------------------------------------------
# 1) LO/OM/LFHW specs (exactly from your plot_fig_3.py)
# -----------------------------------------------------------------------------
CLR: Dict[str, str] = {
    "A-n23": "#D443FF",
    "A-log": "#43FF76",
    "A'": "#7143FF",
    "ABB": "#43CCFF",
    "2+1": "#FF435D",
    "UMDA": "#FF9643",
    "1+ll": "#FFF943",
}
MKR: Dict[str, str] = {
    "A-n23": "o",
    "A-log": "v",
    "A'": "s",
    "ABB": "^",
    "2+1": "P",
    "UMDA": "d",
    "1+ll": "x",
}
LO_SPECS: List[Dict[str, Any]] = []


def add_lo(
    key: str,
    label: str,
    alg: str,
    bench: str,
    desc: str,
    axis: int,
    norm: Optional[str],
):
    LO_SPECS.append(
        {
            "algorithm": alg,
            "benchmark": bench,
            "param_desc": desc,
            "label": label,
            "axis": axis,
            "color": CLR[key],
            "marker": MKR[key],
            "norm": norm,
        }
    )


for bench, ax in [("LO", 0), ("OM", 1), ("LFHW", 2)]:
    base_norm = "n2" if ax == 0 else "nlogn"
    add_lo(
        "A-n23", r"$DEGA_{Proof},\;\lambda=n^{2/3}$", "DEGA", bench, "lamb=n^(2/3)", ax, base_norm
    )
    add_lo(
        "A-log", r"$DEGA_{Proof},\;\lambda=\log n$", "DEGA", bench, "lamb=log(n)", ax, base_norm
    )
    add_lo("A'", r"$DEGA$", "DEGA_A", bench, "", ax, base_norm)
    add_lo("ABB", r"$DEGA_{LO}$", "DEGA_B", bench, "", ax, base_norm)
    add_lo("2+1", r"$(2+1)$-GA", "TPOGA", bench, "", ax, base_norm)
    add_lo("UMDA", r"UMDA", "UMDA", bench, "lamb=sqrt(n)*log(n)", ax, base_norm)
    add_lo("1+ll", r"$(1+\lambda,\lambda)$-GA", "OPLLGA", bench, "", ax, base_norm)

# -----------------------------------------------------------------------------
# 2) MIVS-median specs (from your plot_mivs.py)
# -----------------------------------------------------------------------------


MIVS_SPECS: List[Dict[str, Any]] = []


def add_mivs(key: str, label: str, alg: str, bench: str, desc: str, axis: int):
    MIVS_SPECS.append(
        {
            "algorithm": alg,
            "benchmark": bench,
            "param_desc": desc,
            "label": label,
            "axis": axis,
            "color": CLR[key],
            "marker": MKR[key],
            "norm": lambda n: n**2,
        }
    )


# only one axis → 0
add_mivs("1+ll", r"$(1+\lambda,\lambda)$-GA", "OPLLGA", "MIVS", "lamb=sqrt(log(n))", 0)
add_mivs("ABB", r"$A_{LO}$", "DEGA_B", "MIVS", "", 0)
add_mivs("A-n23", r"$DEGA_{Proof},\lambda=n^{2/3}$", "DEGA_Limit", "MIVS", "lamb=n^(2/3)", 0)
add_mivs("UMDA", "UMDA", "UMDA", "MIVS", "", 0)
add_mivs("A'", r"$DEGA$", "DEGA_A", "MIVS", "", 0)
add_mivs("A-log", r"$DEGA_{Proof},\lambda=\log n$", "DEGA_Limit", "MIVS", "lamb=log(n)", 0)
add_mivs("2+1", r"$(2+1)$-GA", "TPOGA", "MIVS", "", 0)


# -----------------------------------------------------------------------------
# 3) build the 4-column figure
# -----------------------------------------------------------------------------
fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=(18, 4), sharey=False)

# LO/OM/LFHW → mean panel
plot_lo = Plotter(data_dir="data", run_name="Algo-comp-2")  # <— adjust!
plot_lo.plot_any(
    LO_SPECS,
    axes=[ax0, ax1, ax2],
    loglog=True,
    aggregator="mean",
    plot_std=False,
    xlabel=r"$n$",
    ylabel=[
        r"$\bar T$ / $n^{2}$",
        r"$\bar T$ / $n\ \log n$",
        r"$\bar T$ / $n\ \log n)$",
    ],
    title=[r"$LO$", r"$OM$", r"$LFHW$"],
    legend_axes=[0],
    legend_ncol={0: 2},
)


def add_invisible_margin(ax: plt.Axes, factor: float = 1.5) -> None:
    """
    Add an invisible point at (x_max, factor*y_max) to push the y–limit up.
    """
    x_max = ax.get_xlim()[1]
    y_max = ax.get_ylim()[1]
    ax.plot([x_max], [y_max * factor], alpha=0, marker="None")  # totally invisible


# for each axis where the legend sits inside:
add_invisible_margin(ax0, 0.009)  # left panel

# MIVS → median panel
plot_mivs = Plotter(data_dir="data", run_name="MIVS-4")  # <— adjust!
plot_mivs.plot_any(
    MIVS_SPECS,
    axes=[ax3],
    loglog=True,
    aggregator="median",
    plot_std=False,
    xlabel=r"$n$",
    ylabel=r"$\tilde T$ / $n^2$",
    title=[r"$MIVS$ (Median)"],
    legend_axes=[],
)

if leg := ax3.get_legend():
    leg.remove()

import matplotlib.ticker as mticker

# 1) grab & down-sample the old major ticks
xt = ax3.get_xticks()
xt2 = xt[1::2]

# 2) remove all of Matplotlib's own locators & formatters
ax3.xaxis.set_major_locator(mticker.NullLocator())
ax3.xaxis.set_minor_locator(mticker.NullLocator())

# 3) install your fixed, halved tick positions
ax3.xaxis.set_major_locator(mticker.FixedLocator(xt2))

# 4) install a plain ScalarFormatter (no scientific notation, no offset text)
sf = mticker.ScalarFormatter(useOffset=False)
sf.set_scientific(False)
ax3.xaxis.set_major_formatter(sf)

# 5) hide any leftover offset text just in case
ax3.xaxis.get_offset_text().set_visible(False)

plt.tight_layout()
plt.savefig("./plots/merged_fig5_6.pdf", bbox_inches="tight")
plt.show()
