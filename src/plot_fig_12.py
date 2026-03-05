#!/usr/bin/env python3
# JUMP statistical tests in a 1x3 layout (JUMP2/JUMP3/JUMP4), styled like Fig. 7.

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.stats import rankdata, mannwhitneyu
from visualize import Plotter


def holm_bonferroni(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    p_corr = np.zeros(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        p_corr[idx] = min(1.0, running_max)
    reject = p_corr <= alpha
    return reject, p_corr


DATA_DIR = "data"
RUN_NAME = "JUMP_BENCHMARK"
BENCHMARKS = ["JUMP2", "JUMP3", "JUMP4"]

# Keep algorithm set/labels consistent with updated jump plots.
SPECS = [
    (r"$(2+1)$-GA", "TPOGA", ""),
    (r"$DEGA_+$", "DEGA_A", ""),
    (r"$DEGA_{LO}$", "DEGA_B", ""),
    (r"$DEGA^{\lambda=\log n}$", "DEGA", "lamb=log(n)"),
    (r"$DEGA^{\lambda=n^{2/3}}$", "DEGA", "lamb=n^(2/3)"),
]

plotter = Plotter(DATA_DIR, RUN_NAME)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), sharey=False)

for col, bench in enumerate(BENCHMARKS):
    ax = axes[col]
    all_samples = []
    labels = []
    max_n = None

    for label, alg, filt in SPECS:
        runs_all = plotter.get_runs(alg, bench)
        runs = [r for r in runs_all if filt in r["metadata"]["description"]]
        if not runs:
            continue
        max_n_run = max(runs, key=lambda r: r["n"])
        max_n = max_n_run["n"] if max_n is None else max(max_n, max_n_run["n"])
        samples = np.asarray(max_n_run["evals"], float)
        all_samples.append(samples)
        labels.append(label)

    if not all_samples:
        ax.set_axis_off()
        continue

    # Order by median runtime (same idea as Fig. 7/9).
    medians = np.array([np.median(s) for s in all_samples])
    order = np.argsort(rankdata(medians))
    all_samples = [all_samples[i] for i in order]
    labels = [labels[i] for i in order]
    n_algorithms = len(all_samples)

    # Pairwise tests + Holm correction.
    pairs = [(i, j) for i in range(n_algorithms) for j in range(i + 1, n_algorithms)]
    pvals = []
    for i, j in pairs:
        _, p = mannwhitneyu(
            all_samples[i], all_samples[j], alternative="two-sided", method="auto"
        )
        pvals.append(p)
    rej, pvals_corr = holm_bonferroni(pvals, alpha=0.05)

    # Boxplot (Fig. 7 style).
    ax.boxplot(
        all_samples,
        labels=labels,
        showmeans=False,
        medianprops={"color": "#7143FF", "linewidth": 1.8},
    )
    ax.set_title(f"{bench.replace('JUMP', r'$JUMP_')}, " + r"n=20$")
    ax.set_ylabel(r"$T$" if col == 0 else "")
    ax.tick_params(axis="x", rotation=20)

    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    formatter.set_useOffset(False)
    ax.yaxis.set_major_formatter(formatter)

    # Fig. 7-style significance bars.
    y_max_global = max(float(s.max()) for s in all_samples)
    y_min_global = min(float(s.min()) for s in all_samples)
    y_range = y_max_global - y_min_global
    offset = 0.1 * y_range
    bar_heights = np.zeros(n_algorithms)

    for (i, j), sig, pval in zip(pairs, rej, pvals_corr):
        x1, x2 = i + 1, j + 1
        top_i = all_samples[i].max()
        top_j = all_samples[j].max()
        y = max(top_i, top_j, bar_heights[i], bar_heights[j]) + offset

        if not sig:
            # Slightly lower the rightmost red bar for visual clarity.
            if i == n_algorithms - 2 and j == n_algorithms - 1:
                y = y - 0.04 * y_range
            ax.plot([x1, x2], [y, y], color="#FF435D", lw=2)
            ax.text(
                (x1 + x2) / 2,
                y + 0.01 * y_range,
                r"$p = $" + f"{pval:.4f}",
                ha="center",
                va="bottom",
                color="#FF435D",
                fontsize=8,
            )
        elif pval > 0.001:
            ax.plot([x1, x2], [y, y], color="#FF9643", lw=2)
            ax.text(
                (x1 + x2) / 2,
                y + 0.01 * y_range,
                r"$p = $" + f"{pval:.4f}",
                ha="center",
                va="bottom",
                color="#FF9643",
                fontsize=8,
            )

        bar_heights[i] = y
        bar_heights[j] = y

plt.tight_layout()
out = "./plots/JUMP_BENCHMARK_STAT_TEST.pdf"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, format="pdf", dpi=300, transparent=True)
plt.show()
