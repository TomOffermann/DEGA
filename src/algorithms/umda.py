from algorithm import *
import numpy as np


class UMDA(Algorithm):
    def __init__(self, n, lmbd, mu):
        """
        Initialize the Univariate Marginal Distribution Algorithm (UMDA).

        Args:
            n (int): Length of the binary string.
            mu (int): Number of top individuals to select as elites.
            lmbd (int): Size of the population (number of individuals sampled per generation).
        """
        super().__init__(n=n, lmbd=lmbd, mu=mu)
        self.n = n
        self.mu = mu
        self.lmbd = lmbd

        # Initialize with uniform probabilities
        self.probabilities = np.ones(n) * 0.5

    def run(self, problem, optimum, max_evals, eps=0):
        cnt = 0
        best_fitness = -1e10
        while cnt < max_evals:
            # Sample lambda individuals from the current probability distribution.
            population = np.array(
                [np.random.rand(self.n) < self.probabilities for _ in range(self.lmbd)]
            ).astype(int)
            fitness_values = np.array(
                [problem(individual) for individual in population]
            )
            cnt += self.lmbd

            # Select the top mu individuals (ties are broken by the order given in np.argsort).
            elite_indices = np.argsort(fitness_values)[-self.mu :]
            elites = population[elite_indices]

            # Update the probabilities based on the selected elites.
            self.probabilities = np.mean(elites, axis=0)

            # Clip probabilities to the range [1/n, 1 - 1/n] for numerical stability.
            min_prob = 1 / self.n
            max_prob = 1 - 1 / self.n
            self.probabilities = np.clip(self.probabilities, min_prob, max_prob)

            # Best fitness within current generation.
            best_fitness = fitness_values[elite_indices[-1]]

            if best_fitness >= optimum:
                print("converged")
                return (best_fitness, cnt)

        print("exceeded max iterations", best_fitness)
        return (best_fitness, cnt)

    def __str__(self):
        return f"UMDA(n={self.n}, mu={self.mu}, lambda={self.lmbd})"
