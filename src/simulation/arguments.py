import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from benchmarks import Benchmarks


@dataclass
class Job:
    algorithm: str
    algo_args: Dict[str, Any]
    benchmark_key: str
    n: int
    reps: int
    description: str
    max_evals: int
    budget_description: Optional[str] = None


class Arguments:
    """
    Load a JSON file describing a list of jobs:

    {
      "jobs": [
        {
          "algorithm": "dega_a",
          "algo_args": { "lmbd": 50 },
          "benchmark": "LO",
          "n": 100,
          "reps": 20,
          "description": "dega_a(n=100)",
          "max_evals": 30000,
          "budget_description": "30·n·log(n)"
        }
      ]
    }
    """

    def __init__(self, jobs: List[Job]):
        self.jobs = jobs

    @classmethod
    def load_from_file(cls, path: str) -> "Arguments":
        with open(path, "r") as f:
            data = json.load(f)

        jobs: List[Job] = []
        for entry in data.get("jobs", []):
            key = entry.get("benchmark")
            if key not in Benchmarks.keys():
                raise ValueError(
                    f"Unknown benchmark '{key}', available: {Benchmarks.keys()}"
                )
            n = int(entry.get("n"))
            reps = int(entry.get("reps", 1))
            description = entry.get("description", "")

            # budget override or default
            if "max_evals" in entry:
                max_evals = int(entry["max_evals"])
                budget_description = entry.get("budget_description")
            else:
                bmk = Benchmarks.get(key)
                max_evals = bmk.budget_fn(n)
                budget_description = None

            jobs.append(
                Job(
                    algorithm=entry.get("algorithm"),
                    algo_args=entry.get("algo_args", {}),
                    benchmark_key=key,
                    n=n,
                    reps=reps,
                    description=description,
                    max_evals=max_evals,
                    budget_description=budget_description,
                )
            )

        return cls(jobs)
