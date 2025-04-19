import os
import sys
import json
import hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from algorithms import AlgorithmFactory


class Runner:
    """
    A cache-sensitive, parallel runner for AlgorithmFactory algorithms.
    Uses a single tqdm progress bar without per-job logging spam.
    """

    def __init__(self, data_dir: str = "data", max_workers: int = None):
        self.data_dir = Path(data_dir)
        self.max_workers = max_workers

    def run_jobs(self, jobs: list[dict], force: bool = False) -> list[tuple]:
        """
        Run the given jobs, optionally ignoring cache.

        Args:
            jobs: list of dicts with job specs
            force: if True, re-run all jobs even if cached
        Returns:
            List of (job, status, path_or_error)
        """
        statuses = []
        to_run = []

        # divide cached vs to-run
        for job in jobs:
            key = self._job_key(job)
            path = self._result_path(job, key)
            if not force and path.exists():
                statuses.append((job, "cached", str(path)))
            else:
                to_run.append((job, path))

        n_run = len(to_run)
        print(f"{len(statuses)} cached, {n_run} to run")
        if n_run == 0:
            return statuses

        # execute with a single progress bar
        futures = {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for job, path in to_run:
                futures[executor.submit(self._run_single, job, path)] = (job, path)

            pbar = tqdm(total=len(futures), desc="Jobs", unit="job")
            for fut in as_completed(futures):
                job, path = futures[fut]
                try:
                    fut.result()
                    status = "success"
                except Exception:
                    status = "failed"
                    path = None
                statuses.append((job, status, str(path) if path else None))
                pbar.update(1)
                # show only key info
                postfix = f"{job['algorithm']} n={job['n']} {status}"
                pbar.set_postfix_str(postfix)
            pbar.close()

        return statuses

    def _run_single(self, job: dict, path: Path):
        alg = AlgorithmFactory.create(job["algorithm"], **job["algo_args"])
        problem = job["problem"]
        optimum = job["optimum"]
        max_evals = job["max_evals"]
        reps = job.get("reps", 1)
        description = job.get("description", "")

        # suppress any prints inside alg.run
        devnull = open(os.devnull, "w")
        old_stdout = sys.stdout
        sys.stdout = devnull
        results = []
        try:
            for _ in range(reps):
                best, evals = alg.run(problem, optimum, max_evals)
                results.append({"best": best, "evals": evals})
        finally:
            sys.stdout = old_stdout
            devnull.close()

        # save results + metadata
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            results=results,
            metadata=json.dumps(
                {
                    "algorithm": job.get("algorithm"),
                    "benchmark_name": job.get("benchmark_name"),
                    "n": job.get("n"),
                    "reps": job.get("reps"),
                    "algo_args": job.get("algo_args"),
                    "description": description,
                    "optimum": job.get("optimum"),
                    "max_evals": job.get("max_evals"),
                    "budget_desc": job.get("budget_description"),
                }
            ),
        )

    def _job_key(self, job: dict) -> str:
        filt = {k: job[k] for k in sorted(job) if k != "problem"}
        txt = json.dumps(filt, sort_keys=True, default=str)
        return hashlib.md5(txt.encode()).hexdigest()

    def _result_path(self, job: dict, key: str) -> Path:
        return (
            self.data_dir
            / job.get("algorithm")
            / job.get("benchmark_name")
            / f"{key}.npz"
        )
