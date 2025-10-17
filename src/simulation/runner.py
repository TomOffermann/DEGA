import os
import sys
import json
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any

import numpy as np
from tqdm import tqdm

from algorithms import AlgorithmFactory


class Runner:
    """
    A cache‑sensitive, parallel runner for AlgorithmFactory algorithms.
    • Namespaces results under an optional run_name.
    • Shows one tqdm bar per algorithm so you can watch slow ones in isolation.
    """

    def __init__(
        self, data_dir: str = "data", run_name: str = None, max_workers: int = None
    ):
        self.data_dir = Path(data_dir)
        self.run_name = run_name
        self.max_workers = max_workers

    def _data_root(self) -> Path:
        """Return root folder (data/ or data/<run_name>/)."""
        return self.data_dir / self.run_name if self.run_name else self.data_dir

    def run_jobs(self, jobs: List[dict], force: bool = False) -> List[tuple]:
        """
        Run the given jobs, optionally ignoring cache.

        Args:
            jobs: list of dicts with job specs
            force: if True, re‑run all jobs even if cached
        Returns:
            List of (job, status, path_or_error)
        """
        statuses: List[tuple] = []
        to_run: List[tuple] = []

        # split cached vs to‑run
        for job in jobs:
            key = self._job_key(job)
            path = self._result_path(job, key)
            if not force and path.exists():
                statuses.append((job, "cached", str(path)))
            else:
                to_run.append((job, path))

        print(f"{len(statuses)} cached, {len(to_run)} to run")
        if not to_run:
            return statuses

        # group jobs by algorithm
        alg_groups: Dict[str, List[tuple]] = {}
        for job, path in to_run:
            alg_groups.setdefault(job["algorithm"], []).append((job, path))

        # create one bar per algorithm
        bars: Dict[str, Any] = {}
        for i, (alg, group) in enumerate(sorted(alg_groups.items())):
            bars[alg] = tqdm(
                total=len(group),
                desc=f"{alg}",
                unit="job",
                position=i,
                leave=True,
            )

        # schedule all at once
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._run_single, job, path): (job, path)
                for group in alg_groups.values()
                for job, path in group
            }

            for fut in as_completed(futures):
                job, path = futures[fut]
                try:
                    fut.result()
                    status = "success"
                except Exception as e:
                    status = f"failed:{e!r}"
                    path = None

                statuses.append((job, status, str(path) if path else None))
                bar = bars[job["algorithm"]]
                bar.update(1)
                bar.set_postfix_str(status.split(":", 1)[0])

        # close bars
        for bar in bars.values():
            bar.close()

        return statuses

    def _run_single(self, job: dict, path: Path):
        problem = job["problem"]
        optimum = job["optimum"]
        max_evals = job["max_evals"]
        reps = job.get("reps", 1)
        description = job.get("description", "")

        # suppress internal prints
        devnull = open(os.devnull, "w")
        old_stdout = sys.stdout
        sys.stdout = devnull

        results = []
        try:
            for _ in range(reps):
                # fresh instance per repetition
                alg = AlgorithmFactory.create(job["algorithm"], **job["algo_args"])
                best, evals = alg.run(problem, optimum, max_evals)
                results.append({"best": best, "evals": evals})
        finally:
            sys.stdout = old_stdout
            devnull.close()

        # save results + metadata
        root = self._data_root()
        out = root / job["algorithm"] / job["benchmark_name"]
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            out / f"{self._job_key(job)}.npz",
            results=results,
            metadata=json.dumps(
                {
                    "run_name": self.run_name,
                    "algorithm": job["algorithm"],
                    "benchmark_name": job["benchmark_name"],
                    "n": job["n"],
                    "reps": job["reps"],
                    "algo_args": job["algo_args"],
                    "description": description,
                    "optimum": job["optimum"],
                    "max_evals": job["max_evals"],
                    "budget_desc": job.get("budget_description"),
                }
            ),
        )

    def _job_key(self, job: dict) -> str:
        filt = {k: job[k] for k in sorted(job) if k != "problem"}
        txt = json.dumps(filt, sort_keys=True, default=str)
        return hashlib.md5(txt.encode()).hexdigest()

    def _result_path(self, job: dict, key: str) -> Path:
        root = self._data_root()
        return root / job["algorithm"] / job["benchmark_name"] / f"{key}.npz"
