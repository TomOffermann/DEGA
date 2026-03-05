import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from math import log, sqrt
from algorithms import TPOGA, UMDA, DEGA_Limit, DEGA_B, DEGA_A, OPLLGA
from benchmarks import mivs
from benchmarks.benchmark_wrapper import _mivs_opt

from math import log, sqrt

# ------------------------
# Experiment configuration
# ------------------------
N_LIST = [20, 50, 72, 100]
REPS = 1000
BUDGET_FN = lambda n: int(30 * n * log(n))

# ------------------------
# Helpers
# ------------------------


def make_algos(n):
    """
    Return a list of (label, constructor) so we get a FRESH instance per repetition.
    """
    return [
        (
            f"OPLLGA(sqrt(log n))",
            lambda: OPLLGA(n, int(round(sqrt(log(n)))), int(round(sqrt(log(n))))),
            "#FFF943",
            r"$(1+ (\lambda, \lambda))$-GA",
        ),
        (
            f"UMDA(sqrt(n)·log n)",
            lambda: UMDA(n, int(round(sqrt(n) * log(n))), int(round(log(n)))),
            "#FF9643",
            r"UMDA",
        ),
        ("TPOGA", lambda: TPOGA(n), "#FF435D", r"$(2+1)$-GA"),
        (
            f"DEGA_Limit(n^(2/3))",
            lambda: DEGA_Limit(n, int(round(n ** (2 / 3)))),
            "#D443FF",
            r"$DEGA^{\lambda=n^{2/3}}$",
        ),
        ("DEGA_A", lambda: DEGA_A(n), "#7143FF", r"$DEGA_+$"),
        ("DEGA_B", lambda: DEGA_B(n), "#43CCFF", r"$DEGA_{LO}$"),
        (
            f"DEGA_Limit(log n)",
            lambda: DEGA_Limit(n, int(round(log(n)))),
            "#43FF76",
            r"$DEGA^{\lambda=\log n}$",
        ),
    ]


def runs_to_anytime_grid(runs, n):
    """
    runs: list of tuples (cnts, bests) for one algorithm at fixed n
          where cnts and bests are 1D numpy arrays (monotone increasing cnts,
          bests is cumulative max of fitness)
    Returns: t_grid, q25, q50, q75 (np.ndarrays)
    """
    # Union of all observed evaluation counts across runs
    t_grid = np.unique(np.concatenate([cnts for cnts, _, _, _ in runs]))
    # Build value matrix: each row = one run sampled at t_grid (right-constant step function)
    V = np.full((len(runs), len(t_grid)), np.nan)
    for i, (cnts, bests, clr, desc) in enumerate(runs):
        # For each t in t_grid, take last observed best at evaluation <= t
        idx = np.searchsorted(cnts, t_grid, side="right") - 1
        valid = idx >= 0
        V[i, valid] = n / 2 + 2 - bests[idx[valid]]
    # Robust summary
    q25 = np.nanquantile(V, 0.25, axis=0)
    q50 = np.nanquantile(V, 0.50, axis=0)  # median
    q75 = np.nanquantile(V, 0.75, axis=0)
    return t_grid, q25, q50, q75


# ------------------------
# Run experiments + collect
# ------------------------
# results[n][label] = list of runs; each run is (cnts, bests)
from collections import defaultdict

results = defaultdict(lambda: defaultdict(list))

for n in N_LIST:
    budget = BUDGET_FN(n)
    for label, ctor, color, desc in make_algos(n):
        for r in range(REPS):
            algo = ctor()  # fresh instance every repetition
            best_f, cnt, F = algo.run(mivs, _mivs_opt(n), budget, track_fitness=True)
            if not F:
                continue
            cnts, fits = zip(*F)
            cnts = np.asarray(cnts, dtype=int)
            fits = np.asarray(fits, dtype=float)

            # Keep only within budget, sort by count, and convert fits -> best-so-far
            order = np.argsort(cnts)
            cnts = cnts[order]
            fits = fits[order]
            mask = cnts <= budget
            cnts = cnts[mask]
            fits = fits[mask]
            if cnts.size == 0:
                continue
            bests = np.maximum.accumulate(fits)
            results[n][label].append((cnts, bests, color, desc))

# ------------------------
# Plot: single-page 2x2 grid (up to four n's)
# ------------------------
out_path = "./plots/ANYTIME_MIVS.pdf"

with PdfPages(out_path) as pdf:
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    axes = axs.ravel()

    # Build a consistent color map across all subplots
    all_labels = []
    for n in N_LIST:
        for label in results[n].keys():
            if label not in all_labels:
                all_labels.append(label)

    prop_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not prop_colors:
        # Fallback just in case
        prop_colors = [plt.get_cmap("tab10")(i) for i in range(len(all_labels))]
    color_map = {
        label: prop_colors[i % len(prop_colors)] for i, label in enumerate(all_labels)
    }

    # Plot each n into its own tile (leave extras blank)
    legend_handles = {}
    for i in range(4):
        ax = axes[i]
        if i < len(N_LIST):
            n = N_LIST[i]
            any_plotted = False
            for label, runs in results[n].items():
                if not runs:
                    continue
                t_grid, q25, q50, q75 = runs_to_anytime_grid(runs, n)
                color = runs[0][2]
                lbl = runs[0][3]
                (line,) = ax.plot(t_grid, q50, label=lbl, linewidth=1.5, color=color)
                ax.fill_between(t_grid, q25, q75, alpha=0.15, linewidth=0, color=color)
                legend_handles[lbl] = line
                any_plotted = True

            ax.set_title(r"$MIVS$ Anytime Performance $" + f"(n={n})" + r"$")
            ax.set_xlabel("Fitness evaluations")
            ax.set_ylabel(r"$f_{max} - f_t + 1$")
            ax.set_xscale("log", base=10)
            ax.set_yscale("log", base=10)
            ax.grid(True, alpha=0.3)

            if not any_plotted:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    alpha=0.6,
                )
        else:
            ax.set_axis_off()

    # Shared legend across the bottom
    if legend_handles:
        labels = list(legend_handles.keys())
        handles = [legend_handles[l] for l in labels]
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.03),
            ncol=7,
            fontsize="medium",
            frameon=False,
        )
        fig.tight_layout(rect=[0, 0.07, 1, 1])  # leave room for legend
    else:
        fig.tight_layout()

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"Wrote {out_path}")
