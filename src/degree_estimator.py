#!/usr/bin/env python3
"""
lo_degree_estimator.py
----------------------------------------------------------
Estimate the power-law exponent  a  in  T(n) ≈ n^a
for the LO data contained in  data/LO-Plot-1-7500/…

• Set SKIP = k  if you want to discard the k smallest n-values
  before the log–log regression is performed.  (SKIP = 0 ⇒ use all.)
"""

from __future__ import annotations
import numpy as np
from visualize import Plotter  # ← your plotting helper (already installed)

# ─── configuration ────────────────────────────────────────────────────
DATA_DIR = "data"
#RUN_NAME = "LO-Plot-1-7500"
RUN_NAME = "Algo-comp-2"
SKIP = 1  # number of leading points to ignore (0 keeps all)

# (print-label , algorithm , benchmark , substring that identifies the run)
SPECS = [
    #("A  λ=(n log n)^{2/3}", "DEGA", "LO", "(n*log(n))**(2/3)"),
    ("A  λ=n^{2/3}", "DEGA", "LO", "lamb=n^(2/3)"),
    #("A  λ=√n", "DEGA", "LO", "lamb=sqrt(n)"),
    #("A  λ=√{n log n}", "DEGA", "LO", "lamb=sqrt(n*log(n))"),
    #("A  λ=2", "DEGA", "LO", "lamb=2"),
    #("A  λ=n^{1/3}", "DEGA", "LO", "lamb=n^(1/3)"),
    ("A  λ=ln(n)", "DEGA", "LO", "lamb=log(n)"),
]


# ─── utility -----------------------------------------------------------
def geom_slope(ns: np.ndarray, ts: np.ndarray) -> float:
    """least-squares slope in log–log space"""
    return np.polyfit(np.log(ns), np.log(ts), 1)[0]


# ─── main --------------------------------------------------------------
plotter = Plotter(DATA_DIR, RUN_NAME)
print(f"\nEstimating exponents for run set “{RUN_NAME}”  (skip ={SKIP})\n")

for label, alg, bench, filt in SPECS:
    runs_all = plotter.get_runs(alg, bench)
    runs = [r for r in runs_all if filt in r["metadata"]["description"]]

    if not runs:
        print(f"{label:25s}  –  no matching runs")
        continue

    # collect n and mean-runtime –– drop the first SKIP entries
    ns, ts = [], []
    for r in sorted(runs, key=lambda r: r["n"])[SKIP:]:
        ns.append(r["n"])
        ts.append(np.asarray(r["evals"], float).mean())

    if len(ns) < 2:
        print(f"{label:25s}  –  too few data points after skipping")
        continue

    a = geom_slope(np.asarray(ns), np.asarray(ts))
    print(f"{label:25s}  a ≈ {a:7.3f}")
