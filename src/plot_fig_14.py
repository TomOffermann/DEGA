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

from algorithms import DEGA_Diversity_Plots
from benchmarks import leading_ones

from math import log
import matplotlib.pyplot as plt

colors = ["#FFF943", "#FF9643", "#FF435D", "#D443FF", "#7143FF", "#43CCFF", "#43FF76"]


N = 100
ls = [
    {"l": int((N * log(N)) ** (2 / 3)), "label": "(n\ln n)^{2/3}", "color": "#FF9643"},
    {"l": int(N ** (2 / 3)), "label": "n^{2/3}", "color": "#FF435D"},
    {"l": 2, "label": "2", "color": "#D443FF"},
    {"l": int(N ** (1 / 2)), "label": "\sqrt{n}", "color": "#7143FF"},
    {"l": int(N ** (1 / 3)), "label": "n^{1/3}", "color": "#43CCFF"},
]
it = 10000
cnt = 0

(figure, axis) = plt.subplots(1, 1)

plt.title(r"$(2+1)$-$DEGA$($n = 100, \lambda_i$)")

figure.set_size_inches(10, 4)

data = []
over_all_min = 2**29
om = 2**29
for l in ls:
    dega = DEGA_Diversity_Plots(l["l"], N)

    D = []
    min_eval = over_all_min
    me = om
    for i in range(it):
        (f, e, d) = dega.run(leading_ones, N, me+100)
        print(f"Run {i+1}/{it}, Lambda {cnt+1}/{len(ls)}")
        D.append(d)
        min_eval = min(min_eval, len(d))
        me = min(me, e)
    over_all_min = min(over_all_min, min_eval)
    om = min(om, me)
    cnt += 1

    D_new = [0] * min_eval
    print(min_eval, len(D_new))

    for eval in range(min_eval):
        for i in range(it):
            D_new[eval] += D[i][eval]
        D_new[eval] /= it
    data.append(D_new)

cnt = 0
for l in ls:
    (ln,) = axis.plot(
        range(over_all_min),
        data[cnt][:over_all_min],
        label=r"$\lambda_" + f"{cnt+1}" + r" = $" + r"$" + l["label"] + r"$",
        color=l["color"],
    )
    axis.annotate(
        "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"[(cnt) % 20],
        xy=(range(over_all_min)[-1], data[cnt][:over_all_min][-1]),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=14,
        ha="left",
        va="center",
        color=ln.get_color(),
    )
    cnt += 1

plt.xlabel(r"$t$")
plt.ylabel(r"$\bar d(t)$")
plt.legend(frameon=False, ncol=2, handlelength=2.2, columnspacing=1.0)
plt.savefig("./plots/DIVERSITY_DEGA_DIFFERENT_LAMBDA.pdf", bbox_inches="tight")
plt.show()
