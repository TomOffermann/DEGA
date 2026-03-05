from algorithms import DEGA_Diversity_Plots, TPOGA_Diversity_Plots
from benchmarks import leading_ones

import matplotlib.pyplot as plt

N1 = 200
N2 = 200
l = int(N1 ** (2 / 3))

dega = DEGA_Diversity_Plots(l, N1)
tpoga = TPOGA_Diversity_Plots(N2)

(f1, e1, d1) = dega.run(leading_ones, N1, N1**2)
(f2, e2, d2) = tpoga.run(leading_ones, N2, N2**2)

plt.rcParams["text.usetex"] = True
plt.rc("font", family="serif")
figure, axis = plt.subplots(ncols=2)
figure.set_size_inches(12, 3)
# figure.suptitle(
#     "Normalized diversity: $d(t) = H(x^1_t, x^2_t)/(n-f^{min}_t)$", fontsize=16
# )
figure.supxlabel(r"$t$")


axis[1].set_ylabel(r"$d(t)$")
axis[1].plot(
    range(len(d1)),
    d1,
    color="#FF435D",
)
axis[1].plot(
    range(len(d1)),
    [1] * len(d1),
    label=r"$1$",
    color="#868686",
    linewidth=1,
    linestyle="-.",
)
axis[1].plot(
    range(len(d1)),
    [0.5] * len(d1),
    label=r"$1/2$",
    color="#868686",
    linewidth=1,
    linestyle=":",
)
axis[1].plot(
    range(len(d1)),
    [0] * len(d1),
    label=r"$0$",
    color="#868686",
    linewidth=1,
    linestyle="--",
)
axis[1].legend()
axis[1].set_title(r"$(2+1)$-$DEGA$($n = 200, \lambda = n^{2/3}$)")

axis[0].set_ylabel(r"$d(t)$")
axis[0].plot(range(len(d2)), d2, color="#7143FF")
axis[0].plot(
    range(len(d2)),
    [1] * len(d2),
    label=r"$1$",
    color="#868686",
    linewidth=1,
    linestyle="-.",
)
axis[0].plot(
    range(len(d2)),
    [0.5] * len(d2),
    label=r"$1/2$",
    color="#868686",
    linewidth=1,
    linestyle=":",
)
axis[0].plot(
    range(len(d2)),
    [0] * len(d2),
    label=r"$0$",
    color="#868686",
    linewidth=1,
    linestyle="--",
)
axis[0].legend()
axis[0].set_title(r"$(2+1)$-GA($n = 200$)")

plt.savefig("./plots/DIVERSITY_DEGA.pdf", bbox_inches="tight")
plt.show()
