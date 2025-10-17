# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import ScalarFormatter
# from scipy.stats import rankdata, wilcoxon
# from statsmodels.stats.multitest import multipletests
# from visualize import Plotter  # your plotting helper
# import os

# DATA_DIR = "data"
# RUN_NAME = "MIVS-4"

# # Algorithm specifications; benchmark field will be updated in the loop
# SPECS = [
#     (r"$DEGA^{\lambda=log(n)}$", "DEGA_Limit", "LO", "lamb=log(n)"),
#     (r"$DEGA^{\lambda=n^{2/3}}$", "DEGA_Limit", "LO", "lamb=n^(2/3)"),
#     (r"$(1+\lambda, \lambda)-GA$", "OPLLGA", "LO", ""),
#     (r"$DEGA_+$", "DEGA_A", "LO", ""),
#     (r"$(2+1)-GA$", "TPOGA", "LO", ""),
#     (r"$DEGA_LO$", "DEGA_B", "LO", ""),
#     (r"$UMDA$", "UMDA", "LO", "lamb=sqrt(n)*log(n)"),
# ]

# BENCHMARK = "MIVS"
# plotter = Plotter(DATA_DIR, RUN_NAME)

# # Update benchmark in SPECS for current iteration
# specs_for_bench = [(label, alg, BENCHMARK, filt) for (label, alg, _, filt) in SPECS]

# all_samples = []
# labels = []

# for label, alg, bench, filt in specs_for_bench:
#     runs_all = plotter.get_runs(alg, bench)
#     runs = [r for r in runs_all if filt in r["metadata"]["description"]]
#     if not runs:
#         print(f"{label:25s}  –  no matching runs for benchmark {bench}")
#         continue

#     # Find the largest n
#     max_n_run = max(runs, key=lambda r: r["n"])
#     samples = np.asarray(max_n_run["evals"], float)  # all repetitions
#     all_samples.append(samples)
#     labels.append(label)

# n_algorithms = len(all_samples)

# # Compute medians and ranks
# medians = np.array([np.median(s) for s in all_samples])
# ranks = rankdata(medians)

# # Sort by rank (best left)
# order = np.argsort(ranks)
# all_samples = [all_samples[i] for i in order]
# labels = [labels[i] for i in order]
# medians = medians[order]

# print(f"\nBenchmark: {bench}")
# print("Median runtimes:", medians)
# print("Ranks (lower = better):", np.arange(1, n_algorithms + 1))

# # Pairwise Wilcoxon tests
# pairs = [(i, j) for i in range(n_algorithms) for j in range(i + 1, n_algorithms)]
# pvals = []
# for i, j in pairs:
#     stat, p = wilcoxon(all_samples[i], all_samples[j])
#     pvals.append(p)

# # Holm correction
# rej, pvals_corr, _, _ = multipletests(pvals, alpha=0.05, method="holm")
# print("Pairwise Wilcoxon (Holm-corrected):")
# for (i, j), p, sig in zip(pairs, pvals_corr, rej):
#     print(f"{labels[i]} vs {labels[j]}: p={p:.4f}, significant={sig}")

# # Boxplot
# plt.boxplot(all_samples, labels=labels)
# plt.ylabel(r"$T$")
# plt.title(f"{bench}, " + r"$n=$" + f"{max_n_run['n']}")
# plt.xticks(rotation=20)

# # Use scientific notation for y-axis
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-3, 5))  # switch to scientific notation early
# plt.gca().yaxis.set_major_formatter(formatter)

# increment = 0.2
# y_max_global = max([s.max() for s in all_samples])
# y_range = y_max_global - min([s.min() for s in all_samples])
# y_offset_dict = {}  # track the current y-offset for each box index

# offset = 0.1 * y_range  # vertical spacing

# # initialize current bar heights above each box
# bar_heights = np.zeros(n_algorithms)

# for (i, j), sig, pval in zip(pairs, rej, pvals_corr):
#     x1, x2 = i + 1, j + 1
#     top_i = all_samples[i].max()
#     top_j = all_samples[j].max()
#     # place bar just above the boxes and previous bars on these boxes
#     y = max(top_i, top_j, bar_heights[i], bar_heights[j]) + offset
#     # draw bar
#     if not sig:
#         plt.plot([x1, x2], [y, y], color="red", lw=2)
#         plt.text((x1+x2)/2, y + 0.01*y_range, r"$p = $" + f"{pval:.4f}", ha='center', va='bottom', color='red', fontsize=8)

#     elif pval > 0.001:
#         plt.plot([x1, x2], [y, y], color="black", lw=2)
#         plt.text((x1+x2)/2, y + 0.01*y_range, r"$p = $" + f"{pval:.4f}", ha='center', va='bottom', color='black', fontsize=8)

#     # update bar heights only for involved boxes
#     bar_heights[i] = y
#     bar_heights[j] = y

# # for (i, j), sig, pval in zip(pairs, rej, pvals_corr):
# #     x1, x2 = i + 1, j + 1
# #     top_i = all_samples[i].max()
# #     top_j = all_samples[j].max()
    
# #     # base height = max of box tops
# #     base_y = max(top_i, top_j)
    
# #     # track previous offsets for these boxes
# #     prev_offset = max(y_offset_dict.get(i, 0), y_offset_dict.get(j, 0))
# #     y = base_y + prev_offset + increment * y_range  # add a small offset
    
# #     # update offsets for these boxes
# #     y_offset_dict[i] = y - top_i
# #     y_offset_dict[j] = y - top_j

# #     if not sig:
# #         ax.plot([x1, x2], [y, y], color="red", lw=2)
# #         ax.text((x1+x2)/2, y + 0.02*y_range, r"$p = $" + f"{pval:.4f}", ha='center', va='bottom', color='red', fontsize=8)

# #     elif pval > 0.0009:
# #         ax.plot([x1, x2], [y, y], color="black", lw=2)
# #         ax.text((x1+x2)/2, y + 0.02*y_range, r"$p = $" + f"{pval:.4f}", ha='center', va='bottom', color='black', fontsize=8)


# plt.tight_layout()
# out = "./plots/stat-mivs.pdf"
# os.makedirs(os.path.dirname(out), exist_ok=True)
# plt.savefig(out, format="pdf", dpi=300, transparent=True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.stats import rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests
from visualize import Plotter  # your plotting helper
import os

DATA_DIR = "data"
RUN_NAME = "MIVS-4"

# Algorithm specifications; benchmark field will be updated in the loop
SPECS = [
    (r"$DEGA^{\lambda=log(n)}$", "DEGA_Limit", "LO", "lamb=log(n)"),
    (r"$DEGA^{\lambda=n^{2/3}}$", "DEGA_Limit", "LO", "lamb=n^(2/3)"),
    (r"$(1+\lambda, \lambda)-GA$", "OPLLGA", "LO", ""),
    (r"$DEGA_+$", "DEGA_A", "LO", ""),
    (r"$(2+1)-GA$", "TPOGA", "LO", ""),
    (r"$DEGA_LO$", "DEGA_B", "LO", ""),
    (r"$UMDA$", "UMDA", "LO", "lamb=sqrt(n)*log(n)"),
]

BENCHMARK = "MIVS"
plotter = Plotter(DATA_DIR, RUN_NAME)

# Update benchmark in SPECS for current iteration
specs_for_bench = [(label, alg, BENCHMARK, filt) for (label, alg, _, filt) in SPECS]

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

# Pairwise Wilcoxon tests
pairs = [(i, j) for i in range(n_algorithms) for j in range(i + 1, n_algorithms)]
pvals = []
for i, j in pairs:
    stat, p = wilcoxon(all_samples[i], all_samples[j])
    pvals.append(p)

# Holm correction
rej, pvals_corr, _, _ = multipletests(pvals, alpha=0.05, method="holm")
print("Pairwise Wilcoxon (Holm-corrected):")
for (i, j), p, sig in zip(pairs, pvals_corr, rej):
    print(f"{labels[i]} vs {labels[j]}: p={p:.4f}, significant={sig}")

# Boxplot
plt.boxplot(all_samples, labels=labels)
plt.ylabel(r"$T$")
plt.title(f"{bench}, " + r"$n=$" + f"{max_n_run['n']}")
plt.xticks(rotation=20)

# Use scientific notation for y-axis
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-3, 5))  # switch to scientific notation early
plt.gca().yaxis.set_major_formatter(formatter)

# --- Non-overlapping significance bars ---
y_max_global = max([s.max() for s in all_samples])
y_min_global = min([s.min() for s in all_samples])
y_range = y_max_global - y_min_global

bar_spacing = 0.05 * y_range  # vertical gap between bars
bar_heights = np.zeros(n_algorithms)  # track max bar level per algorithm index
y_max_bar = y_max_global

for (i, j), sig, pval in zip(pairs, rej, pvals_corr):
    x1, x2 = i + 1, j + 1
    top_i = all_samples[i].max()
    top_j = all_samples[j].max()

    # find max occupied height across the span [i, j]
    current_max = max(bar_heights[i:j+1])
    base_y = max(top_i, top_j, current_max)

    # place new bar above that
    y = base_y + bar_spacing

    # draw bar
    color = "black" if sig else "#FF435D"
    if not sig or (sig and pval > 0.001):
      plt.plot([x1, x2], [y, y], color=color, lw=2)
      #plt.text((x1+x2)/2, y + 0.01*y_range, r"$p = $" + f"{pval:.4f}",
          #    ha='center', va='bottom', color=color, fontsize=8)

    # update bar heights across the whole span
    for k in range(i, j+1):
        bar_heights[k] = y

    y_max_bar = max(y_max_bar, y)

# Adjust ylim so top bars fit
plt.ylim(y_min_global, y_max_bar + 0.1 * y_range)

plt.tight_layout()
out = "./plots/stat-mivs.pdf"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, format="pdf", dpi=300, transparent=True)
plt.show()
