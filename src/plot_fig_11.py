#!/usr/bin/env python3
# JUMP overview plot: 2x3 grid (Median/Mean) x (JUMP2/JUMP3/JUMP4)

from __future__ import annotations

import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["text.usetex"] = False
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["mathtext.fontset"] = "dejavusans"

from typing import Dict, List, Callable, Optional
from matplotlib import pyplot as plt
from visualize import Plotter
import os


DATA_DIR = "data"
RUN_NAME = "JUMP_BENCHMARK"
BENCHES = ["JUMP2", "JUMP3", "JUMP4"]

# Keep same palette/marker scheme as existing figure scripts.
CLR: Dict[str, str] = {
    "A-n23": "#D443FF",
    "A-log": "#43FF76",
    "A'": "#7143FF",
    "ABB": "#43CCFF",
    "(2+1)": "#FF435D",
}
MKR: Dict[str, str] = {
    "A-n23": "s",
    "A-log": "^",
    "A'": "d",
    "ABB": "v",
    "(2+1)": "P",
}


def k_of(bench: str) -> int:
    return int(bench.replace("JUMP", ""))


def mk_spec(
    key: str,
    label: str,
    alg: str,
    bench: str,
    selector: str,
    axis: int,
    norm: Optional[Callable[[int], float]],
):
    return {
        "algorithm": alg,
        "benchmark": bench,
        "param_desc": selector,
        "label": label,
        "axis": axis,
        "color": CLR[key],
        "marker": MKR[key],
        "norm": norm,
    }


specs: List[Dict] = []
axis_aggr: Dict[int, str] = {}

# 2 rows x 3 columns:
# row 0 = median, row 1 = mean; columns = JUMP2, JUMP3, JUMP4
for col, bench in enumerate(BENCHES):
    k = k_of(bench)
    norm_fn = lambda n, kk=k: n**kk

    ax_med = col
    ax_mean = 3 + col
    axis_aggr[ax_med] = "median"
    axis_aggr[ax_mean] = "mean"

    for ax in (ax_med, ax_mean):
        specs += [
            mk_spec("(2+1)", r"$(2+1)$-GA", "TPOGA", bench, "", ax, norm_fn),
            mk_spec("A'", r"$DEGA_+$", "DEGA_A", bench, "", ax, norm_fn),
            mk_spec("ABB", r"$DEGA_{LO}$", "DEGA_B", bench, "", ax, norm_fn),
            mk_spec(
                "A-log",
                r"$DEGA^{\lambda=\log n}$",
                "DEGA",
                bench,
                "lamb=log(n)",
                ax,
                norm_fn,
            ),
            mk_spec(
                "A-n23",
                r"$DEGA^{\lambda=n^{2/3}}$",
                "DEGA",
                bench,
                "lamb=n^(2/3)",
                ax,
                norm_fn,
            ),
        ]

plotter = Plotter(DATA_DIR, RUN_NAME)
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False)
ax_list = list(axes.ravel())

ylabels = [
    r"$\tilde T(n) / n^2$",
    r"$\tilde T(n) / n^3$",
    r"$\tilde T(n) / n^4$",
    r"$\bar T(n) / n^2$",
    r"$\bar T(n) / n^3$",
    r"$\bar T(n) / n^4$",
]

titles = [r"$JUMP_2$", r"$JUMP_3$", r"$JUMP_4$", "", "", ""]

plotter.plot_any(
    specs,
    axes=ax_list,
    loglog=True,
    plot_std=False,
    axis_aggregators=axis_aggr,
    xlabel=r"$n$",
    ylabel=ylabels,
    title=titles,
    legend_axes=[0],   # one legend only
    legend_ncol={0: 2},
)

# Row labels as overall y-axis descriptors.
fig.subplots_adjust(right=0.95, wspace=0.26, hspace=0.28)
fig.text(0.955, 0.73, "Median", rotation=270, va="center", ha="left", fontsize=12)
fig.text(0.955, 0.28, "Mean", rotation=270, va="center", ha="left", fontsize=12)

out = "./plots/JUMP_BENCHMARK.pdf"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, format="pdf", dpi=300, transparent=True)
plt.show()
