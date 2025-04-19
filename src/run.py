import argparse

from simulation.arguments import Arguments
from simulation.runner import Runner

from benchmarks import Benchmarks


def main():
    parser = argparse.ArgumentParser(
        description="Parallel, cache‑aware runner for your GA experiments."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="jobs.json",
        help="Path to JSON file listing all jobs (default: jobs.json)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: all cores)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force rerun of all jobs, ignoring cache",
    )
    args = parser.parse_args()

    cfg_path = args.config
    ans = input(f"Load jobs from '{cfg_path}'? [Y/n] ").strip().lower()
    if ans and ans[0] == "n":
        cfg_path = input("Enter alternative config file path: ").strip()

    arguments = Arguments.load_from_file(cfg_path)

    # build the concrete jobs list
    jobs = []
    for job in arguments.jobs:
        bm = Benchmarks.get(job.benchmark_key)
        jobs.append(
            {
                "algorithm": job.algorithm,
                "algo_args": job.algo_args,
                "benchmark_name": bm.name,
                "problem": bm.problem,
                "optimum": bm.optimum_fn(job.n),
                "max_evals": job.max_evals,
                "reps": job.reps,
                "description": job.description,
                "n": job.n,
                "budget_description": job.budget_description,
            }
        )

    runner = Runner(data_dir="data", max_workers=args.workers)
    summary = runner.run_jobs(jobs, force=args.force)

    print(f"\n{'ALG':<12} {'BM':<4} {'N':<4} {'MAX_E':<8} {'STATUS':<12} DESC")
    for job, status, info in summary:
        max_e = job.get("max_evals")
        desc = job.get("description")
        print(
            f"{job['algorithm']:<12} {job['benchmark_name']:<4} {job['n']:<4} {max_e:<8} {status:<12} {desc}"
        )


if __name__ == "__main__":
    main()
