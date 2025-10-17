from algorithms.algorithm import *
from algorithms.algorithm_factory import *
from util import *
import random
import numpy as np
from math import log


@AlgorithmFactory.register("DEGA_B")
class DEGA_B(Algorithm):
    def __str__(self):
        return f"(2+1)-DEGA_B(n={self.n}, chi={self.chi}) [limited to u={self.u}]"

    def __init__(self, n, u=None):
        """
        Initialize the DEGA_B.

        Args:
            n (int): Length of the binary string.
            u (int, optional): Number of crossover tries
        """
        self.n = n
        self.chi = 1.0
        if u == None:
            self.u = int(10 * log(n))
        else:
            self.u = u

    def run(self, problem, optimum, max_evals, track_fitness=False):
        x_1 = np.random.randint(0, 2, size=self.n)
        x_2 = 1 - x_1

        f_1 = problem(x_1)
        f_2 = problem(x_2)

        cnt = 2

        F = []

        while cnt < max_evals:
            if track_fitness:
                F.append((cnt, max(f_1, f_2)))
            # Check Convergence
            if max(f_1, f_2) >= optimum:
                print("converged")
                return (max(f_1, f_2), cnt, F) if track_fitness else (max(f_1, f_2), cnt)

            if random.random() < 0.5:  # Mutate
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
                x_1, x_2, f_1, f_2 = select_population_alter_parent(
                    other,
                    parent,
                    offspring,
                    f_other,
                    f_parent,
                    f_off,
                )
            else:
                y = biased_crossover(x_1, x_2, 0.5)

                if f_1 > f_2:
                    x_1, x_2 = x_2, x_1
                    f_1, f_2 = f_2, f_1

                f_y = problem(y)
                cnt += 1

                if f_y > f_1:
                    for _ in range(int(self.u)):
                        offspring = biased_crossover(x_1, y, 1 / 2)
                        f_off = problem(offspring)
                        cnt += 1

                        if f_off > f_1:
                            y = offspring
                            f_y = f_off
                    x_1 = y
                    f_1 = f_y
        print("exceeded max iterations", max(f_1, f_2))
        return (max(f_1, f_2), cnt, F) if track_fitness else (max(f_1, f_2), cnt)
