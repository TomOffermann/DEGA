from algorithms.algorithm import *
from algorithms.algorithm_factory import *
from util import *
import random
import numpy as np

@AlgorithmFactory.register('DEGA')
class DEGA(Algorithm):
    def __init__(self, lmbd, n):
        """
        Initialize the standard DEGA (Section 2).

        Args:
            n (int): Length of the binary string.
            lamb (int): Lambda parameter of the algorithm
        """
        super().__init__(n=n, lmbd=lmbd)
        self.n = n
        self.lmbd = lmbd
        self.chi = 1.0

    def __str__(self):
        return f"(2+1)-DEGA(n={self.n}, lambda={self.lmbd}, chi={self.chi})"

    def run(self, problem, optimum, max_evals, eps=0):
        x_1 = np.random.randint(0, 2, size=self.n)
        x_2 = 1 - x_1

        f_1 = problem(x_1)
        f_2 = problem(x_2)

        cnt = 2

        while cnt < max_evals:
            # Check Convergence
            if max(f_1, f_2) >= optimum:
                print("converged")
                return (max(f_1, f_2), cnt)

            if f_1 != f_2:  # Exploitation Phase
                if f_1 > f_2:
                    x_1, x_2 = x_2, x_1
                    f_1, f_2 = f_2, f_1

                off = biased_crossover(x_1, x_2, 1 / self.lmbd)
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
        return (max(f_1, f_2), cnt)
