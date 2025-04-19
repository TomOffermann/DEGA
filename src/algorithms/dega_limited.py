from algorithms.algorithm import *
from algorithms.algorithm_factory import *
from util import *
from math import log
import random
import numpy as np


@AlgorithmFactory.register('DEGA_Limit')
class DEGA_Limit(Algorithm):
    def __init__(self, n, lamb, u=None):
        """
        Initialize the standard DEGA (Section 2) limited to guarantee some mutation from time to time.

        Args:
            n (int): Length of the binary string.
            lmbd (int): Lambda parameter of the algorithm
            u (int, optional): Number that determines the maximum number of exploitation phases without any mutation. We set u = \lambda \log n as a defaul value.
        """
        super().__init__(n=n, lamb=lamb)
        self.n = n
        self.lamb = lamb
        self.chi = 1.0

        if u == None:
            self.u = int(lamb * log(n))
        else:
            self.u = u

    def __str__(self):
        return f"(2+1)-DEGA(n={self.n}, lambda={self.lamb}, chi={self.chi}) [limited to u={self.u}]"

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

            if f_1 != f_2 and l <= self.u:  # Exploitation Phase
                y = biased_crossover(x_1, x_2, 0.5)

                if f_1 > f_2:
                    x_1, x_2 = x_2, x_1
                    f_1, f_2 = f_2, f_1

                f_y = problem(y)
                cnt += 1
                l += 1

                if f_y > f_1:
                    for i in range(int(self.u)):
                        off = biased_crossover(x_1, y, 1 / self.lamb)
                        f_off = problem(off)
                        cnt += 1

                        # Update when improving bit was found
                        if f_off > f_1:
                            l = 0
                            x_1 = off
                            f_1 = f_off
                            break
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
                x_1, x_2, f_1, f_2, l = select_population_limit(
                    other,
                    parent,
                    offspring,
                    f_off,
                    f_other,
                    f_parent,
                    l,
                )

        print("exceeded max iterations", max(f_1, f_2))
        return (max(f_1, f_2), cnt)
