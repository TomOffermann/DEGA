import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from matplotlib import pyplot as plt

# A 7‑color template
COLORS = [
    "#FFF943",  # yellow
    "#FF9643",  # orange
    "#FF435D",  # red
    "#D443FF",  # magenta
    "#7143FF",  # purple
    "#43CCFF",  # cyan
    "#43FF76",  # green
]


class Plotter:
    """
    Easy plotting of cached simulation runs in ./data.

    Example usage:
        p = Plotter(data_dir="data")
        p.plot_evals_vs_n("DEGA_A", "LO", aggregator="mean", loglog=True)
        plt.show()
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def get_runs(
        self,
        algorithm: str,
        benchmark: str,
    ) -> List[Dict[str, Any]]:
        """
        Load all .npz files under data/<algorithm>/<benchmark>/
        Returns a list of dict with keys:
          - n: problem size
          - evals: list of eval counts per rep
          - metadata: full metadata dict
        """
        folder = self.data_dir / algorithm / benchmark
        files = sorted(folder.glob("*.npz"))
        runs = []
        for path in files:
            arr = np.load(path, allow_pickle=True)
            # metadata stored as JSON string
            meta_json = arr["metadata"].tolist()
            meta = json.loads(meta_json)
            results = arr["results"]
            # results should be an array of dicts
            # convert element-wise to list of dicts
            if isinstance(results, np.ndarray):
                results_list = [r.item() if isinstance(r, np.ndarray) else r for r in results]
            else:
                results_list = list(results)
            evals = [int(r["evals"]) for r in results_list]
            runs.append({"n": int(meta["n"]), "evals": evals, "metadata": meta})
        return runs

    def plot_evals_vs_n(
        self,
        algorithm: str,
        benchmark: str,
        aggregator: str = "mean",
        loglog: bool = False,
        ax: Optional[plt.Axes] = None,
        color: Optional[str] = None,
        label: Optional[str] = None,
    ) -> plt.Axes:
        """
        Plot evals vs n for one algorithm/benchmark.

        aggregator: 'mean' or 'median'
        loglog: whether to use log–log scaling
        """
        runs = self.get_runs(algorithm, benchmark)
        # group by n
        runs = sorted(runs, key=lambda x: x['n'])
        ns = [r['n'] for r in runs]
        values = []
        for r in runs:
            arr = np.array(r['evals'], dtype=float)
            if aggregator == 'mean':
                values.append(arr.mean())
            elif aggregator == 'median':
                values.append(np.median(arr))
            else:
                raise ValueError("aggregator must be 'mean' or 'median'")

        if ax is None:
            fig, ax = plt.subplots()

        if color is None:
            color = COLORS[0]
        if label is None:
            label = f"{algorithm}-{benchmark} ({aggregator})"

        if loglog:
            ax.loglog(ns, values, label=label, color=color)
        else:
            ax.plot(ns, values, label=label, color=color)

        ax.set_xlabel("n")
        ax.set_ylabel(f"f-evals ({aggregator})")
        ax.set_title(f"{algorithm} on {benchmark}")
        ax.legend()
        return ax

    def plot_multiple(
        self,
        specs: List[Dict[str, Any]],
        aggregator: str = "mean",
        loglog: bool = False,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot multiple series on one axes.

        specs: list of dict with keys:
          - algorithm (str)
          - benchmark (str)
          - label (str, optional)
          - color (str, optional)
        """
        if ax is None:
            fig, ax = plt.subplots()

        for idx, spec in enumerate(specs):
            alg = spec['algorithm']
            bm  = spec['benchmark']
            lbl = spec.get('label', f"{alg}-{bm}")
            clr = spec.get('color', COLORS[idx % len(COLORS)])
            self.plot_evals_vs_n(
                alg,
                bm,
                aggregator=aggregator,
                loglog=loglog,
                ax=ax,
                color=clr,
                label=lbl
            )

        return ax
