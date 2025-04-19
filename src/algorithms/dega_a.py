from algorithms.algorithm import *
from algorithms.algorithm_factory import *
from util import *
import random
import numpy as np
from math import log

@AlgorithmFactory.register('DEGA_A')
class DEGA_A:
    def __str__(self):
        return f"(2+1)-DEGA_A(n={self.n}, chi={self.chi})"

    def __init__(self, n):
        """
        Initialize the DEGA_A. Notice that no lambda value is required
        As we use the hamming distance to adapt the parameter dynamically.

        Args:
            n (int): Length of the binary string
        """
        self.n = n
        self.chi = 1.0

    def run(self, problem, optimum, max_evals):
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
                x_1, x_2, f_1, f_2 = select_population(
                    other, parent, offspring, f_other, f_parent, f_off
                )
            else:
                y = biased_crossover(x_1, x_2, 0.5)

                if f_1 > f_2:
                    x_1, x_2 = x_2, x_1
                    f_1, f_2 = f_2, f_1

                f_y = problem(y)
                cnt += 1

                if f_y > f_1:
                    hamm_dist = np.count_nonzero(f_y != f_1)
                    for _ in range(int(hamm_dist * log(self.n))):
                        off = biased_crossover(x_1, y, 1 / hamm_dist)
                        f_off = problem(off)
                        cnt += 1

                        # Update when improving bit was found
                        if f_off > f_1:
                            x_1 = off
                            f_1 = f_off
                            break
        print("exceeded max iterations", max(f_1, f_2))
        return (max(f_1, f_2), cnt)
