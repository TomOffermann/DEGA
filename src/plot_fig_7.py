import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.stats import rankdata, mannwhitneyu
from visualize import Plotter  # your plotting helper
import os


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
RUN_NAME = "MAIN_BENCHMARK"

# Algorithm specifications; benchmark field will be updated in the loop
SPECS = [
    (r"$DEGA^{\lambda=log(n)}$", "DEGA", "LO", "lamb=log(n)"),
    (r"$DEGA^{\lambda=n^{2/3}}$", "DEGA", "LO", "lamb=n^(2/3)"),
    (r"$(1+(\lambda, \lambda))-GA$", "OPLLGA", "LO", ""),
    (r"$DEGA_+$", "DEGA_A", "LO", ""),
    (r"$(2+1)-GA$", "TPOGA", "LO", ""),
    (r"$DEGA_{LO}$", "DEGA_B", "LO", ""),
    (r"$UMDA$", "UMDA", "LO", "lamb=sqrt(n)*log(n)"),
]

BENCHMARKS = ["LO", "OM", "LFHW"]
plotter = Plotter(DATA_DIR, RUN_NAME)

fig, axes = plt.subplots(
    2,2, figsize=(12,8)
)

for a_idx, bench in enumerate(BENCHMARKS):
    # Update benchmark in SPECS for current iteration
    specs_for_bench = [(label, alg, bench, filt) for (label, alg, _, filt) in SPECS]

    all_samples = []
    labels = []

    for label, alg, bench, filt in specs_for_bench:
        runs_all = plotter.get_runs(alg, bench)
        runs = [r for r in runs_all if filt in r["metadata"]["description"]]
        if not runs:
            print(f"{label:25s}  –  no matching runs for benchmark {bench}")
            continue

        # Find the largest n
        max_n_run = max(runs, key=lambda r: r["n"])
        samples = np.asarray(max_n_run["evals"], float)  # all repetitions
        all_samples.append(samples)
        labels.append(label)

    n_algorithms = len(all_samples)
    if n_algorithms == 0:
        continue

    # Compute medians and ranks
    medians = np.array([np.median(s) for s in all_samples])
    ranks = rankdata(medians)

    # Sort by rank (best left)
    order = np.argsort(ranks)
    all_samples = [all_samples[i] for i in order]
    labels = [labels[i] for i in order]
    medians = medians[order]

    print(f"\nBenchmark: {bench}")
    print("Median runtimes:", medians)
    print("Ranks (lower = better):", np.arange(1, n_algorithms + 1))

    # Pairwise Wilcoxon rank-sum (Mann-Whitney U) tests
    pairs = [(i, j) for i in range(n_algorithms) for j in range(i + 1, n_algorithms)]
    pvals = []
    for i, j in pairs:
        _, p = mannwhitneyu(
            all_samples[i], all_samples[j], alternative="two-sided", method="auto"
        )
        pvals.append(p)

    # Holm correction
    rej, pvals_corr = holm_bonferroni(pvals, alpha=0.05)
    print("Pairwise Wilcoxon rank-sum / Mann-Whitney U (Holm-corrected):")
    for (i, j), p, sig in zip(pairs, pvals_corr, rej):
        print(f"{labels[i]} vs {labels[j]}: p={p:.4f}, significant={sig}")

    # Boxplot
    a_idxi = a_idx // 2
    a_idxj = a_idx % 2
    ax = axes[a_idxi, a_idxj]
    ax.boxplot(
        all_samples,
        labels=labels,
        showmeans=False,
        medianprops={
            "color": "#7143FF",
            "linewidth": 1.8,
        },
    )
    ax.set_ylabel(r"$T$" if a_idx == 0 else "")
    ax.set_title(r"$" + f"{bench}, " + r"n=" + f"{max_n_run['n']}" + r"$")
    ax.tick_params(axis="x", rotation=20)

    # Use scientific notation for y-axis
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3, 5))  # switch to scientific notation early
    ax.yaxis.set_major_formatter(formatter)

    increment = 0.2
    y_max_global = max([s.max() for s in all_samples])
    y_range = y_max_global - min([s.min() for s in all_samples])
    y_offset_dict = {}  # track the current y-offset for each box index

    offset = 0.1 * y_range  # vertical spacing

    # initialize current bar heights above each box
    bar_heights = np.zeros(n_algorithms)

    for (i, j), sig, pval in zip(pairs, rej, pvals_corr):
        x1, x2 = i + 1, j + 1
        top_i = all_samples[i].max()
        top_j = all_samples[j].max()
        # place bar just above the boxes and previous bars on these boxes
        y = max(top_i, top_j, bar_heights[i], bar_heights[j]) + offset
        # draw bar
        if not sig:
            ax.plot([x1, x2], [y, y], color="#FF435D", lw=2)
            ax.text((x1+x2)/2, y + 0.01*y_range, r"$p = $" + f"{pval:.4f}", ha='center', va='bottom', color="#FF435D", fontsize=8)

        elif pval > 0.001:
            ax.plot([x1, x2], [y, y], color="#FF9643", lw=2)
            ax.text((x1+x2)/2, y + 0.01*y_range, r"$p = $" + f"{pval:.4f}", ha='center', va='bottom', color="#FF9643", fontsize=8)

        # update bar heights only for involved boxes
        bar_heights[i] = y
        bar_heights[j] = y
    
axes[1, 1].axis('off')
plt.tight_layout()
out = "./plots/MAIN_BENCHMARK_STAT_TEST.pdf"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, format="pdf", dpi=300, transparent=True)
plt.show()
