# plot_fig_3.py  ─────────────────────────────────────────────────────────
from visualize import Plotter
import matplotlib.pyplot as plt
from math import log, sqrt  # sqrt used in the UMDA reference
import os
from typing import List, Dict, Optional

# ── colour / marker lookup tables keyed by a SHORT internal key ─────────
CLR: Dict[str, str] = {
    "A-n23": "#D443FF",
    "A-log": "#43FF76",
    "A'": "#7143FF",
    "ABB": "#43CCFF",
    "(2+1)": "#FF435D",
    "UMDA": "#FF9643",
    "(1+ll)": "#FFF943",
}
MKR: Dict[str, str] = {
    "A-n23": "o",
    "A-log": "v",
    "A'": "s",
    "ABB": "^",
    "(2+1)": "P",
    "UMDA": "d",
    "(1+ll)": "x",
}

# ── helper that appends a single spec dict to “specs” ───────────────────
specs: List[Dict] = []


def add(
    key: str,
    pretty: str,
    alg: str,
    bench: str,
    param_desc: str,
    axis: int,
    norm: Optional[str] = None,
) -> None:
    """`key` drives colour / marker; `pretty` is the legend label."""
    specs.append(
        dict(
            algorithm=alg,
            benchmark=bench,
            param_desc=param_desc,
            label=pretty,
            axis=axis,
            color=CLR[key],
            marker=MKR[key],
            norm=norm,
        )
    )


# ── build the spec list ─────────────────────────────────────────────────
for bench, ax in [("LO", 0), ("OM", 1), ("LFHW", 2)]:
    base_norm = "n2" if ax == 0 else "nlogn"

    # DEGA: two λ-variants
    add(
        "A-n23", r"$DEGA^{\lambda=n^{2/3}}$", "DEGA", bench, "lamb=n^(2/3)", ax, base_norm
    )
    add("A-log", r"$DEGA^{\lambda=\log n}$", "DEGA", bench, "lamb=log(n)", ax, base_norm)

    # companions
    add("A'", r"$DEGA_+$", "DEGA_A", bench, "", ax, base_norm)
    add("ABB", r"$DEGA_{LO}$", "DEGA_B", bench, "", ax, base_norm)
    add("(2+1)", r"$(2+1)$-GA", "TPOGA", bench, "", ax, base_norm)
    add("UMDA", r"UMDA", "UMDA", bench, "lamb=sqrt(n)*log(n)", ax, base_norm)
    add("(1+ll)", r"$(1+(\lambda,\lambda))$-GA", "OPLLGA", bench, "", ax, base_norm)


# ── plotting ────────────────────────────────────────────────────────────
p = Plotter(data_dir="data", run_name="MAIN_BENCHMARK")

fig, (ax_lo, ax_om, ax_lfhw) = plt.subplots(1, 3, figsize=(14, 3.75), sharey=False)

p.plot_any(
    specs,
    axes=[ax_lo, ax_om, ax_lfhw],
    loglog=True,
    plot_std=False,
    xlabel=r"$n$",
    ylabel=[
        r"$\bar T(n)$ / $n^{2}$",
        r"$\bar T(n)$ / $n\ \ln n$",
        r"$\bar T(n)$ / $n\ \ln n$",
    ],
    title=[r"$LO$", r"$OM$", r"$LFHW$"],
    legend_ncol={0: 2},
    legend_axes=[0],  # single combined legend on the first axis
    ref_legend="below",  # compact reference legend underneath
    axis_norms={0: "n2", 1: "nlogn", 2: "nlogn"},  # axis 2 is left un-normalised
)


def add_invisible_margin(ax: plt.Axes, factor: float = 1.5) -> None:
    """
    Add an invisible point at (x_max, factor*y_max) to push the y–limit up.
    """
    x_max = ax.get_xlim()[1]
    y_max = ax.get_ylim()[1]
    ax.plot([x_max], [y_max * factor], alpha=0, marker="None")  # totally invisible


# for each axis where the legend sits inside:
add_invisible_margin(ax_lo, 0.01)  # left panel

# ── save / show ─────────────────────────────────────────────────────────
out = "./plots/MAIN_BENCHMARK.pdf"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, format="pdf", dpi=300, transparent=True)
plt.show()
