#!/usr/bin/env python3
# plot_mivs.py – adaptive-MIVS runtime (median & mean) on log-log axes
# -------------------------------------------------------------------
from __future__ import annotations

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

from typing import List, Dict
from matplotlib import pyplot as plt
from visualize import Plotter
import math

# ─── where the runs live ─────────────────────────────────────────────
DATA_DIR = "data"
RUN_NAME = "MIVS-4"  # folder created from jobs_mivs_exact.json

# ─── colour / marker palette (short keys) ────────────────────────────
CLR: Dict[str, str] = {
    "1+ll": "#FFF943",
    "ABB": "#43CCFF",
    "A-n23": "#D443FF",
    "A-log": "#43FF76",
    "A'": "#7143FF",
    "UMDA": "#FF9643",
    "2+1": "#FF435D",
}
MKR = {k: "ovs^dxP"[i] for i, k in enumerate(CLR)}


def mk_spec(
    key: str,
    label: str,
    algo: str,
    bench: str,
    selector: str,
    axis: int,
):
    """Compose one spec-dict understood by Plotter.plot_any()."""
    return {
        "algorithm": algo,
        "benchmark": bench,
        "param_desc": selector,
        "label": label,
        "axis": axis,
        "color": CLR[key],
        "marker": MKR[key],
        "norm": lambda n: n**2,
    }


# ─── build spec lists ────────────────────────────────────────────────
specs: List[Dict] = []

# left panel – MEDIANS  (axis 0)
specs += [
    mk_spec(
        "1+ll",
        r"$(1+\lambda,\lambda)$-GA",
        "OPLLGA",
        "MIVS",
        "lamb=sqrt(log(n))",
        axis=0,
    ),
    mk_spec("ABB", r"$DEGA_{LO}$", "DEGA_B", "MIVS", "", axis=0),
    mk_spec(
        "A-n23",
        r"$DEGA^{\lambda=n^{2/3}}$",
        "DEGA_Limit",
        "MIVS",
        "lamb=n^(2/3)",
        axis=0,
    ),
    mk_spec("UMDA", r"UMDA", "UMDA", "MIVS", "lamb=sqrt(n)*log(n)", axis=0),
    mk_spec("A'", r"$DEGA_+$", "DEGA_A", "MIVS", "", axis=0),
    mk_spec("2+1", r"$(2+1)$-GA", "TPOGA", "MIVS", "", axis=0),
    mk_spec(
        "A-log",
        r"$DEGA^{\lambda=\log n}$",
        "DEGA_Limit",
        "MIVS",
        "lamb=log(n)",
        axis=0,
    ),
]

# right panel – MEANS   (axis 1)
specs += [
    mk_spec(
        "1+ll",
        r"$(1+\lambda,\lambda)$-GA",
        "OPLLGA",
        "MIVS",
        "lamb=sqrt(log(n))",
        axis=1,
    ),
    mk_spec("ABB", r"$DEGA_{LO}$", "DEGA_B", "MIVS", "", axis=1),
    mk_spec(
        "A-n23",
        r"$DEGA_{Proof}^{\lambda=n^{2/3}}$",
        "DEGA_Limit",
        "MIVS",
        "lamb=n^(2/3)",
        axis=1,
    ),
    mk_spec("UMDA", r"UMDA", "UMDA", "MIVS", "lamb=sqrt(n)*log(n)", axis=1),
    mk_spec("A'", r"$DEGA$", "DEGA_A", "MIVS", "", axis=1),
    mk_spec("2+1", r"$(2+1)$-GA", "TPOGA", "MIVS", "", axis=1),
    mk_spec(
        "A-log",
        r"$DEGA_{Proof}^{\lambda=\log n}$",
        "DEGA_Limit",
        "MIVS",
        "lamb=log(n)",
        axis=1,
    ),
]

# ─── plotting ────────────────────────────────────────────────────────
plotter = Plotter(DATA_DIR, RUN_NAME)
fig, (ax_med, ax_mean) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

plotter.plot_any(
    specs,
    axes=[ax_med, ax_mean],
    loglog=True,
    plot_std=False,
    axis_aggregators={0: "median", 1: "mean"},  # <-- NEW
    xlabel=r"$n\in[20,100]$",
    ylabel=[r"$\tilde T(n)$ / $n \ \ln n$", r"$\bar T(n)$ / $n \ \ln n$"],
    title=["Median", "Mean"],
    legend_axes=[0],
    legend_ncol={0: 2, 1: 0},  # one column per panel
)


def add_invisible_margin(ax: plt.Axes, factor: float = 1.5) -> None:
    """
    Add an invisible point at (x_max, factor*y_max) to push the y–limit up.
    """
    x_max = ax.get_xlim()[1]
    y_max = ax.get_ylim()[1]
    ax.plot([x_max], [y_max * factor], alpha=0, marker="None")  # totally invisible


# for each axis where the legend sits inside:
add_invisible_margin(ax_med, 1.8)  # left panel


plt.tight_layout()
import os

out = "./plots/mivs.pdf"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, format="pdf", dpi=300, transparent=True)

plt.show()
