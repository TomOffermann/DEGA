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
import matplotlib.pyplot as plt
from math import log

# -------------------------------- prepare figure & Plotter -------------
fig, (ax_lo, ax_om, ax_lfhw) = plt.subplots(1, 3, figsize=(12, 4))
p = Plotter(data_dir="data", run_name="TEST")

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

# -------------------------------- reference curve ----------------------
refs = [
    dict(
        func=lambda n: n**2, label=r"$n^2$", color="gray", linestyle="--", axis=[0]
    )  # only on LO axis
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
    ref_funcs=refs,
    ref_legend="below",  # put compact ref-legend below figure
    title=[r"$LO$", r"$OM$", r"$LFHW$"],
    axis_norms=axis_norms,
)

# save (optional)
import os

out = "./plots/DEGA-Comp.pdf"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, format="pdf", dpi=300, transparent=True)
plt.show()
