from visualize import Plotter
import matplotlib.pyplot as plt

p = Plotter(data_dir="data")

# Single series
ax = p.plot_evals_vs_n("DEGA_A", "LO", aggregator="median", loglog=True)

# Multiple series
specs = [
    {"algorithm":"DEGA_A", "benchmark":"LO", "label":"DEGA_A"},
    {"algorithm":"DEGA_B", "benchmark":"LO", "label":"DEGA_B"},
]
ax2 = p.plot_multiple(specs, aggregator="mean", loglog=True)

plt.show()
