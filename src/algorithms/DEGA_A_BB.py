import random
import numpy as np
from math import log


def biased_crossover(x1, x2, p):
    mask = np.random.rand(len(x1)) < p
    return np.where(mask, x2, x1)


def select_population(other, parent, offspring, f_other, f_parent, f_off):
    if f_off > f_parent:
        return other, offspring, f_other, f_off
    elif f_off == f_parent:
        h_other_parent = np.count_nonzero(other != parent)
        h_other_offspring = np.count_nonzero(other != offspring)

        if h_other_parent > h_other_offspring:
            return other, parent, f_other, f_parent
        elif h_other_parent == h_other_offspring:
            return random.choice(
                [(other, parent, f_other, f_parent), (other, offspring, f_other, f_off)]
            )
        else:
            return other, offspring, f_other, f_off
    else:
        return other, parent, f_other, f_parent


class DEGA_new_iterated:
    def __str__(self):
        return "(2+1)-DEGA"

    def __init__(self, lmbd, n):
        """
        Initialize the DEGA.

        Args:
            n (int): Length of the binary string.
            lamb (int): Lambda of the algorithm
        """
        self.n = n
        self.chi = 1.0
        self.lmbd = lmbd

    def __call__(self, problem, optimum, max_evals):
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
                    other=other,
                    parent=parent,
                    offspring=offspring,
                    f_off=f_off,
                    f_other=f_other,
                    f_parent=f_parent,
                )
            else:
                y = biased_crossover(x_1, x_2, 0.5)

                if f_1 > f_2:
                    x_1, x_2 = x_2, x_1
                    f_1, f_2 = f_2, f_1

                f_y = problem(y)
                cnt += 1

                if f_y > f_1:
                    for _ in range(int(10*log(self.n))):
                        offspring = biased_crossover(x_1, y, 1 / 2)
                        f_off = problem(offspring)
                        cnt += 1

                        if f_off > f_1:
                            y = offspring
                            f_y = f_off
                    x_1 = y
                    f_1 = f_y
        print("exceeded max iterations", max(f_1, f_2))
        return (max(f_1, f_2), cnt)
