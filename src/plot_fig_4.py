# plot_fig_3.py  ─────────────────────────────────────────────────────────
from visualize import Plotter
import matplotlib.pyplot as plt
from math import log, sqrt  # sqrt used in the UMDA reference
import os
from typing import List, Dict, Optional

# ── colour / marker lookup tables keyed by a SHORT internal key ─────────
CLR: Dict[str, str] = {
    "A-n23": "#D443FF",
    "(1+ll)": "#FFF943",
}
MKR: Dict[str, str] = {
    "A-n23": "o",
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
for bench, ax in [("OM", 0)]:
    base_norm = "nlogn"

    add(
        "A-n23", r"$A\ (\lambda=n^{2/3})$", "DEGA", bench, "lamb=n^(2/3)", ax, base_norm
    )
    add("(1+ll)", r"$(1+\lambda,\lambda)$-GA", "OPLLGA", bench, "", ax, base_norm)

# ── reference curves (only axis 0, which is /n²) ────────────────────────
refs = [
    dict(
        func=lambda n: n**2,
        label=r"$n^{2}$",
        color="gray",
        linestyle="--",
        axis=[0],
    )
]

# ── plotting ────────────────────────────────────────────────────────────
p = Plotter(data_dir="data", run_name="Test-1ll")

fig, ax = plt.subplots(figsize=(4, 3))

p.plot_any(
    specs,
    axes=[ax],
    loglog=False,
    plot_std=True,
    xlabel=r"$n$",
    ylabel="",
    title="OM",
    legend_axes=[0],  # single combined legend on the first axis
    axis_norms={0: "nlogn"},  # axis 2 is left un-normalised
)

# ── save / show ─────────────────────────────────────────────────────────
# out = "./plots/fig3_algo_comp.pdf"
# os.makedirs(os.path.dirname(out), exist_ok=True)
# plt.savefig(out, format="pdf", dpi=300, transparent=True)
plt.show()
