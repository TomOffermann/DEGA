from algorithms.algorithm import *
from algorithms.algorithm_factory import *
from util import *
import random
import numpy as np


@AlgorithmFactory.register("DEGA_Diversity_Plots")
class DEGA_Diversity_Plots(Algorithm):
    def __init__(self, lamb, n):
        """
        Initialize the standard DEGA (Section 2).

        Only variation that a run returns
        the diversity for every iteration

        Args:
            n (int): Length of the binary string.
            lamb (int): Lambda parameter of the algorithm
        """
        super().__init__(n=n, lamb=lamb)
        self.n = n
        self.lamb = lamb
        self.chi = 1.0

    def __str__(self):
        return f"(2+1)-DEGA(n={self.n}, lambda={self.lamb}, chi={self.chi})"

    def run(self, problem, optimum, max_evals, eps=0):
        div = []
        x_1 = np.random.randint(0, 2, size=self.n)
        x_2 = 1 - x_1

        f_1 = problem(x_1)
        f_2 = problem(x_2)

        cnt = 2

        while cnt < max_evals:
            div.append(np.count_nonzero(x_1 != x_2) / (self.n - min(f_1, f_2)))

            # Check Convergence
            if max(f_1, f_2) >= optimum:
                print("converged")
                return (max(f_1, f_2), cnt, div)

            if f_1 != f_2:  # Exploitation Phase
                if f_1 > f_2:
                    x_1, x_2 = x_2, x_1
                    f_1, f_2 = f_2, f_1

                off = biased_crossover(x_1, x_2, 1 / self.lamb)
                f_off = problem(off)
                cnt += 1

                if f_off > f_1:
                    x_1 = off
                    f_1 = f_off
            else:  # Diversity Phase
                (parent, other, f_parent, f_other) = (
                    (x_1, x_2, f_1, f_2)
                    if random.random() < 0.5
                    else (x_2, x_1, f_2, f_1)
                )
                offspring = np.copy(parent)

                mask = np.random.rand(self.n) < (self.chi / self.n)
                offspring[mask] = 1 - offspring[mask]

                f_off = problem(offspring)
                cnt += 1
                x_1, x_2, f_1, f_2 = select_population(
                    other, parent, offspring, f_other, f_parent, f_off
                )

        print("exceeded max iterations", max(f_1, f_2))
        return (max(f_1, f_2), cnt, div)
