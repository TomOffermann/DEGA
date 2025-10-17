"""
Plotter – B/W-friendly visualisation
────────────────────────────────────
Public helper →  plot_any()

 • data curves   : solid line + marker
 • reference     : dashed / dot-dash, no marker  (circled indices ①②③…)
 • σ–bars        : slim error-bar whiskers
 • axis / series normalisation: 'n2', 'nlogn', callable, or None
 • a curve may appear on several sub-plots:  axis = int | list[int]

New in this version
───────────────────
 • axis_aggregators : Dict[int,str]  – pick “mean” / “median” per axis
 • legend_ncol      : int | Dict[int,int] – #columns in each axis-legend
"""

from pathlib import Path
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Callable,
    Tuple,
    Union,
    DefaultDict,
    Sequence,
)
from collections import defaultdict
import json, re, numpy as np
from math import log
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# ─── palettes ───────────────────────────────────────────────────────────
COLORS = ["#FFF943", "#FF9643", "#FF435D", "#D443FF", "#7143FF", "#43CCFF", "#43FF76"]
MARKERS = ["o", "s", "^", "d", "x", "+", "v", "P"]


def _cyc(lst: Sequence[Any], i: int):
    return lst[i % len(lst)]


def _norm(spec: Union[str, Callable[[int], float], None]):
    if spec is None:
        return None
    if spec == "n2":
        return lambda n: n**2
    if spec == "nlogn":
        return lambda n: n * log(n)
    if callable(spec):
        return spec
    raise ValueError(f"unknown normaliser {spec!r}")


# =======================================================================
class Plotter:
    _circ = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

    # ------------------------------------------------------------------
    def __init__(self, data_dir="data", run_name: Optional[str] = None):
        self.root = Path(data_dir) / run_name if run_name else Path(data_dir)

    # ------------------ disk I/O --------------------------------------
    def get_runs(self, alg: str, bench: str) -> List[Dict[str, Any]]:
        folder = self.root / alg / bench
        runs = []
        for f in sorted(folder.glob("*.npz")):
            arr = np.load(f, allow_pickle=True)
            meta = json.loads(arr["metadata"].tolist())
            res = arr["results"]
            if isinstance(res, np.ndarray):
                res = [r.item() if isinstance(r, np.ndarray) else r for r in res]
            runs.append(
                dict(
                    n=int(meta["n"]),
                    evals=[int(r["evals"]) for r in res],
                    metadata=meta,
                )
            )
        runs.sort(key=lambda r: r["n"])
        return runs

    # ------------------ reference helper ------------------------------
    def _draw_ref(self, ax: Axes, xs, rf, nfn, store):
        ys = [rf["func"](x) / (nfn(x) if nfn else 1) for x in xs]
        (ln,) = ax.plot(
            xs,
            ys,
            color=rf.get("color", "gray"),
            linestyle=rf.get("linestyle", "--"),
            linewidth=1.2,
            label="_nolegend_",
            zorder=1,
        )
        store.append(ln)

        # circled index
        if not hasattr(self, "_ref_counter"):
            self._ref_counter = []
        if rf not in self._ref_counter:
            self._ref_counter.append(rf)
        idx = self._ref_counter.index(rf) + 1
        circ = self._circ[idx - 1] if idx <= len(self._circ) else f"[{idx}]"

        ax.annotate(
            circ,
            xy=(xs[-1], ys[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=14,
            ha="left",
            va="center",
            color=ln.get_color(),
        )

    # ==================================================================
    #  MAIN ENTRY POINT
    # ==================================================================
    def plot_any(
        self,
        series: List[Dict[str, Any]],
        *,
        aggregator: str = "mean",
        axis_aggregators: Union[Dict[int, str], None] = None,  # NEW
        axes=None,
        loglog: bool = False,
        ref_funcs=None,
        ref_legend=True,
        legend_axes=None,
        legend_ncol: Union[int, Dict[int, int], None] = None,
        xlabel="n",
        ylabel=None,
        title=None,
        color_order=None,
        plot_std: bool = False,
        xlim=None,
        ylim=None,
        axis_norms=None,
    ):
        # fresh ref counter per figure
        self._ref_counter = []

        axis_norms = axis_norms or {}
        legend_axes = list(legend_axes) if legend_axes else [0]
        axis_aggr = axis_aggregators or {}

        # -------------- set-up axes -----------------------------------
        if axes is None:
            fig, ax0 = plt.subplots()
            axL = [ax0]
        elif isinstance(axes, Axes):
            axL = [axes]
        else:
            axL = list(axes)
        nA = len(axL)

        # -------------- collect series --------------------------------
        curves = []
        for spec in series:
            runs = self.get_runs(spec["algorithm"], spec["benchmark"])
            filt = spec["param_desc"]
            sel = [
                r
                for r in runs
                if (
                    filt(r["metadata"]["description"])
                    if callable(filt)
                    else filt in r["metadata"]["description"]
                )
            ]
            if not sel:
                print("warning: spec matched nothing →", spec)
                continue

            tgt = spec.get("axis", 0)
            tgt = list(tgt) if isinstance(tgt, (list, tuple, set)) else [tgt]

            curves.append(
                (
                    tgt,
                    spec["label"],
                    sel,
                    spec.get("color"),
                    spec.get("marker"),
                    _norm(spec.get("norm")),
                )
            )

        if color_order:
            order = {lab: i for i, lab in enumerate(color_order)}
            curves.sort(key=lambda t: order.get(t[1], len(order)))
        else:
            curves.sort(key=lambda t: t[1])

        xs_ax: DefaultDict[int, set] = defaultdict(set)
        data_h, ref_h = [], []

        # -------------- draw data curves ------------------------------
        for idx, (targets, lab, runs, col, mkr, nfn_spec) in enumerate(curves):
            col = col or _cyc(COLORS, idx + 1)  # skip yellow
            mkr = mkr or _cyc(MARKERS, idx)

            ns, ys, sd = [], [], []
            for r in sorted(runs, key=lambda r: r["n"]):
                n = r["n"]
                arr = np.asarray(r["evals"], float)

                # decide statistic for *first* axis of the curve
                ag = axis_aggr.get(targets[0], aggregator)
                if ag == "mean":
                    v = arr.mean()
                    s = arr.std()
                elif ag == "median":
                    v = np.median(arr)
                    s = np.median(np.abs(arr - v))
                else:
                    raise ValueError(f"unknown aggregator {ag!r}")

                nfn = nfn_spec or _norm(axis_norms.get(targets[0]))
                if nfn:
                    div = nfn(n)
                    v /= div
                    s /= div
                ns.append(n)
                ys.append(v)
                sd.append(s)

            for ax_i in targets:
                if ax_i >= nA:
                    raise IndexError(f"axis index {ax_i} out of range (have {nA})")
                ax = axL[ax_i]
                xs_ax[ax_i].update(ns)

                if plot_std:
                    ax.errorbar(
                        ns,
                        ys,
                        yerr=sd,
                        fmt="none",
                        ecolor=col,
                        elinewidth=0.8,
                        capsize=2,
                    )

                (h,) = (ax.loglog if loglog else ax.plot)(
                    ns,
                    ys,
                    color=col,
                    marker=mkr,
                    markersize=4,
                    linewidth=1.3,
                    linestyle="-",
                    label=lab,
                )
                data_h.append(h)

        # -------------- reference curves ------------------------------
        if ref_funcs:
            for rf in ref_funcs:
                tgt = rf.get("axis", list(range(nA)))
                tgt = [tgt] if isinstance(tgt, int) else list(tgt)
                for ax_i in tgt:
                    xs = (
                        np.geomspace(xlim[0], xlim[1], 200)
                        if xlim
                        else sorted(xs_ax[ax_i])
                    )
                    self._draw_ref(
                        axL[ax_i], xs, rf, _norm(axis_norms.get(ax_i)), ref_h
                    )

        # -------------- per-axis cosmetics & legends ------------------
        def pick(v, i):
            return v[i] if isinstance(v, (list, tuple)) else v

        for i, ax in enumerate(axL):
            ax.set_xscale("log" if loglog else "linear")
            ax.set_yscale("log" if loglog else "linear")
            if xlim:
                ax.set_xlim(xlim)
            if ylim:
                ax.set_ylim(ylim)

            xt = sorted(xs_ax[i])
            ax.set_xticks(xt)
            ax.set_xticklabels(xt, rotation=45)
            ax.set_xlabel(pick(xlabel, i))
            ax.set_ylabel(pick(ylabel or r"$\bar T(n)$", i))
            ax.set_title(pick(title, i))
            ax.grid(True, linestyle=":", alpha=0.5)

            if i in legend_axes:
                handles = [h for h in data_h if h.axes is ax]
                ncol = (
                    legend_ncol.get(i) if isinstance(legend_ncol, dict) else legend_ncol
                )
                ax.legend(
                    fontsize=10,
                    handles=handles,
                    frameon=False,
                    ncol=ncol if ncol else 1,
                    handlelength=2.2,
                    columnspacing=1.0,
                )

        # -------------- external reference legend --------------------
        if ref_legend and ref_h:
            fig = axL[0].get_figure()
            vis_refs = [h for h in ref_h if h.get_label() != "_nolegend_"]
            if vis_refs:
                if ref_legend is True:
                    tgt = axL[legend_axes[0]]
                    tgt.add_artist(
                        tgt.legend(
                            handles=vis_refs,
                            loc="upper left",
                            ncol=len(vis_refs),
                            frameon=False,
                        )
                    )
                else:
                    loc = {"below": (0.5, -0.18), "right": (1.02, 0.5)}.get(
                        ref_legend, ref_legend
                    )
                    fig.legend(
                        handles=vis_refs,
                        loc="center",
                        bbox_to_anchor=loc,
                        ncol=len(vis_refs),
                        frameon=False,
                    )
                    if ref_legend == "below":
                        fig.subplots_adjust(bottom=0.25)

        plt.tight_layout()
        return axL[0] if nA == 1 else axL

    # ------------------ back-compat wrapper ---------------------------
    def plot_specs(self, specs: List[Dict[str, Any]], **kw):
        return self.plot_any(specs, **kw)

    def plot_by_description_params(self, *_, **__):
        raise RuntimeError("plot_by_description_params() was superseded by plot_any().")
