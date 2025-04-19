from algorithms.algorithm import *
from algorithms.algorithm_factory import *
from util import *
import numpy as np


@AlgorithmFactory.register("OPOGA")
class OPOGA(Algorithm):
    def __init__(self, n):
        """
        Initialize the (1+1)-GA.

        Args:
            n (int): Length of the binary string.
        """
        super().__init__(n=n)
        self.n = n
        self.chi = 1.0

    def run(self, problem, optimum, max_evals, eps=0):
        n = self.n
        x = np.random.randint(2, size=self.n)
        f_x = problem(x)
        cnt = 1

        while cnt < max_evals:
            # Check convergence
            if f_x >= optimum:
                print("converged")
                return (f_x, cnt)

            mutation_rate = self.chi / n
            mut = mutate(x, mutation_rate)
            mut_f = problem(mut)
            cnt += 1

            # Update the best solution using fitness comparison.
            # (prefer offspring in ties)
            if mut_f >= f_x:
                x = mut
                f_x = mut_f

        print("exceeded max iterations", f_x)
        return (f_x, cnt)

    def __str__(self):
        return f"(1+1)-GA(n={self.n}, chi={self.chi})"
