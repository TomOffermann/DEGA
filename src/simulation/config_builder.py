import json
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import numpy as np

from benchmarks import Benchmarks
from algorithms import AlgorithmFactory


class JobSuiteBuilder:
    """
    Helper to programmatically build a suite of jobs and write to a JSON file.

    Allows sweeping over different "n" values and resolving any algorithm parameters
    that are functions of n (e.g., lambda n: int(n * log(n))). Supports both linear
    and logarithmic n ranges, and lets you override the evaluation budget (max_evals).

    Example:
        builder = JobSuiteBuilder()
        builder.add_range_sweep(
            algorithm='dega_a',
            algo_args={'chi':1.0},
            benchmark_key='LO',
            n_start=20,
            n_end=100,
            range_type='log',
            num=10,
            reps=20,
            budget=lambda n: int(50 * n * np.log(n)),
            param_descriptions={'chi':'1.0'},
            budget_description='50·n·log(n)'
        )
        builder.write('jobs.json')
    """

    def __init__(self):
        self._jobs: List[Dict[str, Any]] = []

    def add_sweep(
        self,
        algorithm: str,
        algo_args: Dict[str, Union[int, Callable[[int], Any]]],
        benchmark_key: str,
        n_values: List[int],
        reps: int = 1,
        budget: Union[int, Callable[[int], int]] = None,
        param_descriptions: Dict[str, str] = None,
        budget_description: str = None,
    ) -> None:
        """
        Add a sweep over explicit n_values.

        Parameters:
        - algorithm: name as registered in AlgorithmFactory.
        - algo_args: dict of parameter -> int or callable(n) -> int.
        - benchmark_key: key from Benchmarks.
        - n_values: list of problem sizes.
        - reps: number of repetitions per job.
        - budget: optional int or callable(n)->int to override max_evals.
        - param_descriptions: optional dict for human-readable param names.
        - budget_description: optional human description of the budget formula.
        """
        if algorithm not in AlgorithmFactory.available():
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Available: {AlgorithmFactory.available()}"
            )
        if benchmark_key not in Benchmarks.keys():
            raise ValueError(
                f"Unknown benchmark '{benchmark_key}'. Available: {Benchmarks.keys()}"
            )

        alg_cls = AlgorithmFactory._registry[algorithm]
        sig = inspect.signature(alg_cls.__init__)
        valid_params = set(sig.parameters) - {"self"}
        for param in algo_args:
            if param not in valid_params:
                raise ValueError(
                    f"Parameter '{param}' not valid for algorithm '{algorithm}'. Valid: {sorted(valid_params)}"
                )

        for n in n_values:
            # resolve algorithm args
            # inspect the real __init__ signature
            sig = inspect.signature(alg_cls.__init__)
            resolved_args: Dict[str, Any] = {}
            for pname, param in sig.parameters.items():
                if pname == 'self':
                    continue
                if pname == 'n':
                    resolved_args['n'] = int(n)
                elif pname in algo_args:
                    raw = algo_args[pname]
                    val = raw(n) if callable(raw) else raw
                    # if val is a float but the default isn't float, cast to int
                    default = param.default
                    if isinstance(val, float) and not isinstance(default, float):
                        val = int(val)
                    resolved_args[pname] = val
                # otherwise leave it out so __init__ uses its own default


            # determine budget
            if budget is None:
                bmk = Benchmarks.get(benchmark_key)
                resolved_budget = bmk.budget_fn(n)
                budget_desc = None
            else:
                resolved_budget = budget(n) if callable(budget) else budget
                if budget_description:
                    budget_desc = budget_description
                elif callable(budget) and hasattr(budget, "__name__"):
                    budget_desc = budget.__name__
                else:
                    budget_desc = str(budget)

            # human-readable description
            parts = [f"n={n}"]
            for k, v in algo_args.items():
                if param_descriptions and k in param_descriptions:
                    desc = param_descriptions[k]
                elif (
                    callable(v) and hasattr(v, "__name__") and v.__name__ != "<lambda>"
                ):
                    desc = v.__name__
                elif callable(v):
                    desc = "<expr>"
                else:
                    desc = str(v)
                parts.append(f"{k}={desc}")
            if budget_desc:
                parts.append(f"budget={budget_desc}")
            description = f"{algorithm}({', '.join(parts)})"

            self._jobs.append(
                {
                    "algorithm": algorithm,
                    "algo_args": resolved_args,
                    "benchmark": benchmark_key,
                    "n": int(n),
                    "reps": int(reps),
                    "max_evals": int(resolved_budget),
                    "budget_description": budget_desc,
                    "description": description,
                }
            )

    def add_log_sweep(
        self,
        algorithm: str,
        algo_args: Dict[str, Union[int, Callable[[int], Any]]],
        benchmark_key: str,
        n_start: int,
        n_end: int,
        num: int,
        reps: int = 1,
        budget: Union[int, Callable[[int], int]] = None,
        param_descriptions: Dict[str, str] = None,
        budget_description: str = None,
    ) -> None:
        """
        Add a sweep over n values spaced logarithmically between n_start and n_end.
        """
        n_vals = np.logspace(
            np.log10(n_start), np.log10(n_end), num=num, dtype=int
        ).tolist()
        n_vals = sorted(set(n_vals))
        self.add_sweep(
            algorithm=algorithm,
            algo_args=algo_args,
            benchmark_key=benchmark_key,
            n_values=n_vals,
            reps=reps,
            budget=budget,
            param_descriptions=param_descriptions,
            budget_description=budget_description,
        )

    def add_linear_sweep(
        self,
        algorithm: str,
        algo_args: Dict[str, Union[int, Callable[[int], Any]]],
        benchmark_key: str,
        n_start: int,
        n_end: int,
        step: int,
        reps: int = 1,
        budget: Union[int, Callable[[int], int]] = None,
        param_descriptions: Dict[str, str] = None,
        budget_description: str = None,
    ) -> None:
        """
        Add a sweep over n values from n_start to n_end in steps of size step.
        """
        n_vals = list(range(n_start, n_end + 1, step))
        self.add_sweep(
            algorithm=algorithm,
            algo_args=algo_args,
            benchmark_key=benchmark_key,
            n_values=n_vals,
            reps=reps,
            budget=budget,
            param_descriptions=param_descriptions,
            budget_description=budget_description,
        )

    def add_range_sweep(
        self,
        algorithm: str,
        algo_args: Dict[str, Union[int, Callable[[int], Any]]],
        benchmark_key: str,
        n_start: int,
        n_end: int,
        range_type: str = "log",
        num: int = None,
        step: int = None,
        reps: int = 1,
        budget: Union[int, Callable[[int], int]] = None,
        param_descriptions: Dict[str, str] = None,
        budget_description: str = None,
    ) -> None:
        """
        Add a sweep over n values between n_start and n_end.

        range_type: 'log' (requires num) or 'linear' (requires step).
        num: number of points if 'log'.
        step: increment if 'linear'.
        """
        if range_type == "log":
            if num is None:
                raise ValueError("`num` required for 'log'")
            n_vals = np.logspace(
                np.log10(n_start), np.log10(n_end), num=num, dtype=int
            ).tolist()
        elif range_type == "linear":
            if step is None:
                raise ValueError("`step` required for 'linear'")
            n_vals = list(range(n_start, n_end + 1, step))
        else:
            raise ValueError("range_type must be 'log' or 'linear'")
        n_vals = sorted(set(n_vals))
        self.add_sweep(
            algorithm=algorithm,
            algo_args=algo_args,
            benchmark_key=benchmark_key,
            n_values=n_vals,
            reps=reps,
            budget=budget,
            param_descriptions=param_descriptions,
            budget_description=budget_description,
        )

    def jobs(self) -> List[Dict[str, Any]]:
        """Return the list of built job dicts."""
        return self._jobs

    def write(self, path: Union[str, Path]) -> None:
        """Write the jobs to a JSON file at `path`."""
        obj = {"jobs": self._jobs}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(obj, f, indent=2)


class AlgorithmInspector:
    """
    Utility to introspect available algorithms and their constructor parameters.

    Example:
        AlgorithmInspector.list_algorithms()
        AlgorithmInspector.get_signature('dega_a')
        AlgorithmInspector.get_defaults('dega_b')
    """

    @staticmethod
    def list_algorithms() -> List[str]:
        return AlgorithmFactory.available()

    @staticmethod
    def get_signature(algorithm: str) -> inspect.Signature:
        if algorithm not in AlgorithmFactory.available():
            raise ValueError(f"Unknown algorithm '{algorithm}'")
        alg_cls = AlgorithmFactory._registry[algorithm]
        return inspect.signature(alg_cls.__init__)

    @staticmethod
    def get_defaults(algorithm: str) -> Dict[str, Any]:
        sig = AlgorithmInspector.get_signature(algorithm)
        return {
            name: param.default
            for name, param in sig.parameters.items()
            if name != "self" and param.default is not inspect._empty
        }
