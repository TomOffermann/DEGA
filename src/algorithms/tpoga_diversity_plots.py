from algorithms.algorithm import *
from algorithms.algorithm_factory import *
import random
import numpy as np
from util import *


@AlgorithmFactory.register("TPOGA_Diversity_Plots")
class TPOGA_Diversity_Plots(Algorithm):
    def __init__(self, n):
        """
        Initialize standard (2+1)-GA.

        A run returns the diversity for every iteration,
        further we break ties in favor of hamm dist

        Args:
            n (int): Length of the binary string.
        """
        super().__init__(n=n)
        self.n = n
        self.chi = 1.0

    def ham(u, v):
        return int(np.count_nonzero(u != v))

    def run(self, problem, optimum, max_evals, eps=0):
        div = []

        # Initialize a population of two individuals.
        self.population = [np.random.randint(2, size=self.n) for _ in range(2)]
        self.fitness = [problem(ind) for ind in self.population]
        cnt = 2
        best_fitness = max(self.fitness)

        while cnt < max_evals:
            div.append(
                np.count_nonzero(self.population[0] != self.population[1])
                / (self.n - min(*self.fitness))
            )

            # Determine crossover or copy strategy.
            if random.random() < 0.5:
                offspring = uniform_crossover(self.population[0], self.population[1])
            else:
                offspring = random.choice(self.population)

            # Mutation phase: Mutate the offspring.
            mutation_rate = self.chi / self.n
            offspring = mutate(offspring, mutation_rate)
            offspring_fitness = problem(offspring)
            cnt += 1

            # Selection phase: Form a candidate pool including the offspring.
            candidates = self.population + [offspring]
            candidate_fitness = self.fitness + [offspring_fitness]

            # Sort candidates (prefer offspring in ties by favoring index==2).
            sorted_indices = sorted(range(3), key=lambda i: -candidate_fitness[i])

            if (
                candidate_fitness[sorted_indices[1]]
                == candidate_fitness[sorted_indices[2]]
            ):
                if candidate_fitness[sorted_indices[0]] > candidate_fitness[
                    sorted_indices[1]
                ] and np.count_nonzero(
                    candidates[sorted_indices[0]] != candidates[sorted_indices[2]]
                ) > np.count_nonzero(
                    candidates[sorted_indices[0]] != candidates[sorted_indices[1]]
                ):
                    # swap last two:
                    tmp = sorted_indices[2]
                    sorted_indices[2] = sorted_indices[1]
                    sorted_indices[1] = tmp
                if (
                    candidate_fitness[sorted_indices[0]]
                    == candidate_fitness[sorted_indices[1]]
                ):
                    # Find the pair with highest hamming dist:
                    a = candidates[sorted_indices[0]]
                    b = candidates[sorted_indices[1]]
                    c = candidates[sorted_indices[2]]
                    pairs = sorted(
                        [(a, b, [0,1,2]), (a, c, [0,2,1]), (b, c, [1,2,0])],
                        key=lambda pair: -np.count_nonzero(pair[0] != pair[1]),
                    )[0][2]
                    n_idx = [0,0,0]
                    for i in range(3):
                        n_idx[i] = sorted_indices[pairs[i]]
                    sorted_indices = n_idx
                    
            self.population = [
                candidates[sorted_indices[0]],
                candidates[sorted_indices[1]],
            ]
            self.fitness = [
                candidate_fitness[sorted_indices[0]],
                candidate_fitness[sorted_indices[1]],
            ]
            best_fitness = max(self.fitness)

            # Check for convergence.
            if best_fitness >= optimum:
                print("converged")
                return (best_fitness, cnt, div)

        print("exceeded max iterations", best_fitness)
        return (best_fitness, cnt, div)

    def __str__(self):
        return f"(2+1)-GA(n={self.n}, chi={self.chi})"
