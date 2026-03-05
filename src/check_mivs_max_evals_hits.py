#!/usr/bin/env python3
"""
Inspect how often MIVS runs hit the max-evaluations budget.

Default: scans data/MIVS_BENCHMARK and prints:
1) per algorithm summary
2) per (n, algorithm) summary
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

import numpy as np


def load_npz(npz_path: Path):
    arr = np.load(npz_path, allow_pickle=True)
    md = json.loads(arr["metadata"].tolist())
    raw = arr["results"]
    if isinstance(raw, np.ndarray):
        raw = [r.item() if isinstance(r, np.ndarray) else r for r in raw]
    return md, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Base data directory")
    parser.add_argument("--run-name", default="MIVS_BENCHMARK", help="Run folder name")
    args = parser.parse_args()

    root = Path(args.data_dir) / args.run_name
    if not root.exists():
        raise FileNotFoundError(f"Run folder not found: {root}")

    by_algo = defaultdict(lambda: {"jobs": 0, "runs": 0, "hits": 0})
    by_n_algo = defaultdict(lambda: {"jobs": 0, "runs": 0, "hits": 0})

    files = sorted(root.glob("*/*/*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files found in {root}")

    for npz in files:
        md, results = load_npz(npz)
        algo = md.get("algorithm", npz.parts[-3])
        n = int(md.get("n", -1))
        max_evals = int(md["max_evals"])

        evals = np.asarray([int(r["evals"]) for r in results], dtype=int)
        hits = int(np.sum(evals >= max_evals))
        runs = int(evals.size)

        s = by_algo[algo]
        s["jobs"] += 1
        s["runs"] += runs
        s["hits"] += hits

        t = by_n_algo[(n, algo)]
        t["jobs"] += 1
        t["runs"] += runs
        t["hits"] += hits

    print(f"Scanned: {root}")
    print(f"Jobs: {len(files)}")
    print()

    print("Per algorithm")
    print("algorithm   jobs   runs   hit_max  hit_rate")
    for algo, s in sorted(by_algo.items()):
        rate = s["hits"] / s["runs"] if s["runs"] else 0.0
        print(f"{algo:10s}  {s['jobs']:4d}  {s['runs']:6d}  {s['hits']:7d}  {rate:7.2%}")

    print()
    print("Per n + algorithm")
    print("n    algorithm   jobs   runs   hit_max  hit_rate")
    for (n, algo), s in sorted(by_n_algo.items()):
        rate = s["hits"] / s["runs"] if s["runs"] else 0.0
        print(f"{n:4d} {algo:10s}  {s['jobs']:4d}  {s['runs']:6d}  {s['hits']:7d}  {rate:7.2%}")


if __name__ == "__main__":
    main()

