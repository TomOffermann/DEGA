#!/usr/bin/env python3
"""
Build experiment jobs for fixed-k JUMP benchmarks.

Default setup follows reviewer/supervisor discussion:
- fixed k in {2,3,4}
- vary n in {10,12,14,16,18,20}
- compare standard (2+1)-GA and DEGA variants
- optional practical set focuses on DEGA_A
"""

import argparse
from math import log

from simulation import JobSuiteBuilder


K_SETTINGS = [("JUMP2", 2), ("JUMP3", 3), ("JUMP4", 4)]


def parse_n_values(raw: str):
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def add_full_alg_set(builder: JobSuiteBuilder, benchmark_key: str, n_values, reps, budget, budget_desc):
    # Standard baseline
    builder.add_sweep(
        algorithm="TPOGA",
        algo_args={},
        benchmark_key=benchmark_key,
        n_values=n_values,
        reps=reps,
        budget=budget,
        budget_description=budget_desc,
        param_descriptions={},
    )

    # DEGA variants
    builder.add_sweep(
        algorithm="DEGA",
        algo_args={"lamb": lambda n: log(n)},
        benchmark_key=benchmark_key,
        n_values=n_values,
        reps=reps,
        budget=budget,
        budget_description=budget_desc,
        param_descriptions={"lamb": "log(n)"},
    )
    builder.add_sweep(
        algorithm="DEGA",
        algo_args={"lamb": lambda n: n ** (2 / 3)},
        benchmark_key=benchmark_key,
        n_values=n_values,
        reps=reps,
        budget=budget,
        budget_description=budget_desc,
        param_descriptions={"lamb": "n^(2/3)"},
    )
    builder.add_sweep(
        algorithm="DEGA_A",
        algo_args={},
        benchmark_key=benchmark_key,
        n_values=n_values,
        reps=reps,
        budget=budget,
        budget_description=budget_desc,
        param_descriptions={},
    )
    builder.add_sweep(
        algorithm="DEGA_B",
        algo_args={},
        benchmark_key=benchmark_key,
        n_values=n_values,
        reps=reps,
        budget=budget,
        budget_description=budget_desc,
        param_descriptions={},
    )
    builder.add_sweep(
        algorithm="DEGA_Limit",
        algo_args={"lamb": lambda n: log(n)},
        benchmark_key=benchmark_key,
        n_values=n_values,
        reps=reps,
        budget=budget,
        budget_description=budget_desc,
        param_descriptions={"lamb": "log(n)"},
    )
    builder.add_sweep(
        algorithm="DEGA_Limit",
        algo_args={"lamb": lambda n: n ** (2 / 3)},
        benchmark_key=benchmark_key,
        n_values=n_values,
        reps=reps,
        budget=budget,
        budget_description=budget_desc,
        param_descriptions={"lamb": "n^(2/3)"},
    )


def add_pair_alg_set(builder: JobSuiteBuilder, benchmark_key: str, n_values, reps, budget, budget_desc):
    # Small/focused pair for fast convergence scouting.
    builder.add_sweep(
        algorithm="TPOGA",
        algo_args={},
        benchmark_key=benchmark_key,
        n_values=n_values,
        reps=reps,
        budget=budget,
        budget_description=budget_desc,
        param_descriptions={},
    )
    builder.add_sweep(
        algorithm="DEGA",
        algo_args={"lamb": lambda n: log(n)},
        benchmark_key=benchmark_key,
        n_values=n_values,
        reps=reps,
        budget=budget,
        budget_description=budget_desc,
        param_descriptions={"lamb": "log(n)"},
    )


def add_practical_alg_set(builder: JobSuiteBuilder, benchmark_key: str, n_values, reps, budget, budget_desc):
    # Practical comparison requested in review discussion:
    # baseline (2+1)-GA vs practical DEGA variant (DEGA_A).
    builder.add_sweep(
        algorithm="TPOGA",
        algo_args={},
        benchmark_key=benchmark_key,
        n_values=n_values,
        reps=reps,
        budget=budget,
        budget_description=budget_desc,
        param_descriptions={},
    )
    builder.add_sweep(
        algorithm="DEGA_A",
        algo_args={},
        benchmark_key=benchmark_key,
        n_values=n_values,
        reps=reps,
        budget=budget,
        budget_description=budget_desc,
        param_descriptions={},
    )


def main():
    parser = argparse.ArgumentParser(description="Create fixed-k JUMP run configs.")
    parser.add_argument(
        "--n-values",
        default="10,12,14,16,18,20",
        help="Comma-separated n values",
    )
    parser.add_argument("--n-values-k2", default=None, help="Optional n-values override for JUMP2")
    parser.add_argument("--n-values-k3", default=None, help="Optional n-values override for JUMP3")
    parser.add_argument("--n-values-k4", default=None, help="Optional n-values override for JUMP4")
    parser.add_argument("--reps", type=int, default=1000, help="Repetitions per job")
    parser.add_argument(
        "--budget-scale",
        type=float,
        default=5.0,
        help="Budget multiplier c in c*n^k",
    )
    parser.add_argument(
        "--alg-set",
        choices=["full", "pair", "practical"],
        default="full",
        help="Algorithm set: full benchmark set, fast DEGA(log) pair, or practical DEGA_A pair",
    )
    parser.add_argument("--out", default="jobs_jump.json", help="Output json path")
    args = parser.parse_args()

    n_values_default = parse_n_values(args.n_values)
    builder = JobSuiteBuilder()

    for benchmark_key, k in K_SETTINGS:
        if k == 2 and args.n_values_k2:
            n_values = parse_n_values(args.n_values_k2)
        elif k == 3 and args.n_values_k3:
            n_values = parse_n_values(args.n_values_k3)
        elif k == 4 and args.n_values_k4:
            n_values = parse_n_values(args.n_values_k4)
        else:
            n_values = n_values_default

        budget = lambda n, _k=k: int(args.budget_scale * (n ** _k))
        budget_desc = f"{args.budget_scale}*n^{k}"

        if args.alg_set == "full":
            add_full_alg_set(
                builder=builder,
                benchmark_key=benchmark_key,
                n_values=n_values,
                reps=args.reps,
                budget=budget,
                budget_desc=budget_desc,
            )
        elif args.alg_set == "pair":
            add_pair_alg_set(
                builder=builder,
                benchmark_key=benchmark_key,
                n_values=n_values,
                reps=args.reps,
                budget=budget,
                budget_desc=budget_desc,
            )
        else:
            add_practical_alg_set(
                builder=builder,
                benchmark_key=benchmark_key,
                n_values=n_values,
                reps=args.reps,
                budget=budget,
                budget_desc=budget_desc,
            )

    builder.write(args.out)
    print(f"Job-suite written -> {args.out}")


if __name__ == "__main__":
    main()
