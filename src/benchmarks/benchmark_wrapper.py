import numpy as np
import math
from dataclasses import dataclass
from typing import Callable, Dict

from benchmarks.leadingones import *
from benchmarks.lfhw import *
from benchmarks.mivs import *
from benchmarks.onemax import *
from benchmarks.jump import *


@dataclass
class Benchmark:
    """
    Encapsulates a benchmark problem.

    problem: Callable that takes a numpy.ndarray of 0/1 ints and returns a float fitness.
    """

    problem: Callable[[np.ndarray], float]
    name: str
    optimum_fn: Callable[[int], float]
    budget_fn: Callable[[int], int]


class Benchmarks:
    _registry: Dict[str, Benchmark] = {}

    @classmethod
    def register(
        cls,
        key: str,
        name: str,
        problem: Callable[[np.ndarray], float],
        optimum_fn: Callable[[int], float],
        budget_fn: Callable[[int], int],
    ):
        cls._registry[key] = Benchmark(
            problem=problem,
            name=name,
            optimum_fn=optimum_fn,
            budget_fn=budget_fn,
        )

    @classmethod
    def get(cls, key: str) -> Benchmark:
        return cls._registry[key]

    @classmethod
    def keys(cls):
        return list(cls._registry.keys())


# --- helper functions ---


def _mivs_opt(n: int) -> float:
    lookup = {
        20: 8.93,
        22: 10.05,
        24: 10.75,
        26: 11.67,
        28: 12.48,
        30: 13.37,
        34: 15.06,
        36: 15.96,
        40: 17.69,
        42: 18.78,
        46: 20.14,
        50: 21.94,
        56: 24.65,
        60: 26.33,
        66: 28.93,
        72: 31.45,
        78: 34.21,
        84: 36.6,
        92: 40.09,
        100: 43.61,
    }
    return int(round(lookup.get(n, 0.0) - 0.5))


def _default_budget(n: int) -> int:
    # 30 * n * log(n), rounds down
    return int(30 * (n * math.log(n)))

def _jump2(x: np.ndarray) -> float:
    return jump_m(x, 2)


def _jump3(x: np.ndarray) -> float:
    return jump_m(x, 3)


def _jump4(x: np.ndarray) -> float:
    return jump_m(x, 4)

# register all benchmarks
Benchmarks.register("LO", "LO", leading_ones, lambda n: n, _default_budget)
Benchmarks.register("OM", "OM", one_max, lambda n: n, _default_budget)
Benchmarks.register("MIVS", "MIVS", mivs, _mivs_opt, _default_budget)
Benchmarks.register(
    "LFHW", "LFHW", linear_harmonic, lambda n: n * (n + 1) // 2, _default_budget
)
Benchmarks.register("JUMP2", "JUMP2", _jump2, lambda n: n, _default_budget)
Benchmarks.register("JUMP3", "JUMP3", _jump3, lambda n: n, _default_budget)
Benchmarks.register("JUMP4", "JUMP4", _jump4, lambda n: n, _default_budget)
# Backward compatibility: default JUMP = JUMP_4.
Benchmarks.register("JUMP", "JUMP4", _jump4, lambda n: n, _default_budget)
