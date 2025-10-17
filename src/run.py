import argparse
import shutil
from pathlib import Path

from simulation.arguments import Arguments
from benchmarks import Benchmarks
from simulation.runner import Runner


def main():
    parser = argparse.ArgumentParser(
        description="Parallel, cache-aware runner for your GA experiments."
    )
    parser.add_argument(
        "-c", "--config",
        default="jobs.json",
        help="Path to JSON file listing all jobs (default: jobs.json)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: all cores)"
    )
    parser.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force rerun of all jobs, ignoring cache"
    )
    parser.add_argument(
        "-x", "--clean",
        action="store_true",
        help="Remove all cached data for this run and exit"
    )
    parser.add_argument(
        "-n", "--run-name",
        default=None,
        help="Optional name to namespace this batch of runs (creates data/<run-name>/...)"
    )
    args = parser.parse_args()

    if args.clean:
        data_root = Path("data") / args.run_name if args.run_name else Path("data")
        if data_root.exists():
            print(f"Removing {data_root} ...")
            shutil.rmtree(data_root)
            print("Clean complete.")
        else:
            print(f"No cache found at {data_root}")
        return

    cfg_path = args.config
    ans = input(f"Load jobs from '{cfg_path}'? [Y/n] ").strip().lower()
    if ans and ans[0] == "n":
        cfg_path = input("Enter alternative config file path: ").strip()

    arguments = Arguments.load_from_file(cfg_path)

    jobs = []
    for job in arguments.jobs:
        bm = Benchmarks.get(job.benchmark_key)
        jobs.append({
            "algorithm":       job.algorithm,
            "algo_args":       job.algo_args,
            "benchmark_name":  bm.name,
            "problem":         bm.problem,
            "optimum":         bm.optimum_fn(job.n),
            "max_evals":       job.max_evals,
            "reps":            job.reps,
            "description":     job.description,
            "n":               job.n,
            "budget_description": job.budget_description,
        })

    runner = Runner(data_dir="data", run_name=args.run_name, max_workers=args.workers)
    summary = runner.run_jobs(jobs, force=args.force)

    print(f"\n{'ALG':<12} {'BM':<4} {'N':<4} {'MAX_E':<8} {'STATUS':<12} DESC")
    for job, status, _ in summary:
        print(f"{job['algorithm']:<12} {job['benchmark_name']:<4} {job['n']:<4} "
              f"{job['max_evals']:<8} {status:<12} {job['description']}")

if __name__ == "__main__":
    main()