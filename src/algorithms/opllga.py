from algorithms.algorithm import *
from algorithms.algorithm_factory import *
from util import *
import numpy as np

@AlgorithmFactory.register('OPLLGA')
class OPLLGA(Algorithm):
    def __init__(self, n, lamb, chi):
        """
        Initialize the (1+(lambda,lambda))-GA.

        Args:
            n (int): Length of the binary string.
            lamb (int): Lambda parameter of the algorithm.
            chi (float): Mutation scaling factor (mutation probability is chi/n).
        """
        super().__init__(n=n, lamb=lamb, chi=chi)
        self.n = n
        self.lamb = lamb
        self.chi = chi

    def run(self, problem, optimum, max_evals, eps=0):
        n = self.n
        x = np.random.randint(2, size=n)
        f_x = problem(x)
        cnt = 1

        while cnt < max_evals:
            # Check convergence
            if f_x >= optimum:
                print("converged")
                return (f_x, cnt)

            mutation_rate = self.chi / n
            # Generate lambda mutants from the current solution
            mutants = [mutate(x, mutation_rate) for _ in range(self.lamb)]
            mutant_fitness = [problem(mutant) for mutant in mutants]
            cnt += self.lamb

            # Select the best mutant
            best_mutant_idx = np.argmax(mutant_fitness)
            best_mutant = mutants[best_mutant_idx]

            # Recombination phase: generate lambda offspring via crossover
            crossover_rate = 1 / self.chi
            offspring = [
                biased_crossover(x, best_mutant, crossover_rate)
                for _ in range(self.lamb)
            ]
            offspring_fitness = [problem(child) for child in offspring]
            cnt += self.lamb

            # Select the best offspring
            best_offspring_idx = np.argmax(offspring_fitness)
            best_offspring = offspring[best_offspring_idx]

            # Update the best solution if an offspring improves fitness
            if offspring_fitness[best_offspring_idx] > f_x:
                x = best_offspring
                f_x = offspring_fitness[best_offspring_idx]

        print("exceeded max iterations", f_x)
        return (f_x, cnt)

    def __str__(self):
        return f"(1+(l,l))-GA(n={self.n}, lamb={self.lamb}, chi={self.chi})"
