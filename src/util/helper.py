import numpy as np
import random


def mutate(individual, mutation_rate):
    mask = np.random.rand(len(individual)) < mutation_rate
    return np.where(mask, 1 - individual, individual)


def uniform_crossover(parent1, parent2):
    mask = np.random.rand(len(parent1)) < 0.5
    return np.where(mask, parent1, parent2)


def biased_crossover(x1, x2, p):
    mask = np.random.rand(len(x1)) < p
    return np.where(mask, x2, x1)


def select_population(x1, x2, x3, f1, f2, f3):
    """
    Select two individuals (and their fitnesses) among x1, x2, x3 by:
      1. Maximizing the fitness
      2. Breaking fitness-ties by Hamming distance
      3. Picking uniformly at random if still tied

    Args:
        x1, x2, x3 (np.ndarray): candidate bit-strings
        f1, f2, f3 (float): their respective fitnesses

    Returns:
        (a, b, f_a, f_b): chosen pair and their fitnesses,
        with f_a >= f_b.
    """

    # Hamming distance between two bit‑arrays
    def ham(u, v):
        return int(np.count_nonzero(u != v))

    # build list of all pairs
    candidates = []
    for u, v, fu, fv in ((x1, x2, f1, f2), (x1, x3, f1, f3), (x2, x3, f2, f3)):
        score = (max(fu, fv), min(fu, fv))  # lexicographic compare
        d = ham(u, v)
        candidates.append((u, v, fu, fv, score, d))

    # 1) select highest (max, min) pair
    best_score = max(item[4] for item in candidates)
    best = [item for item in candidates if item[4] == best_score]

    # 2) if tie, pick by maximum Hamming distance
    if len(best) > 1:
        max_d = max(item[5] for item in best)
        best = [item for item in best if item[5] == max_d]

    # 3) if still multiple, choose uniformly
    chosen = random.choice(best)
    a, b, fa, fb, _, _ = chosen

    # ensure ordering so fa >= fb
    if fb > fa:
        a, b, fa, fb = b, a, fb, fa

    return a, b, fa, fb


def select_population_limit(x1, x2, x3, f1, f2, f3, l):
    """
    Select two individuals (and their fitnesses) among x1, x2, x3 by:
      1. Maximizing the fitness
      2. Breaking fitness-ties by Hamming distance
      3. Picking uniformly at random if still tied
    Reset new_l to 0 if the chosen pair includes x3 (the offspring), else leave l unchanged.
    Args:
        x1 (ndarray): Candidate 1
        x2 (ndarray): Candidate 2
        x3 (ndarray): Offspring
        f1 (int): f(x1)
        f2 (int): f(x2)
        f3 (int): f(x3)
        l (int): Limit parameter, only used in limited (crossocer iterations) algorithms
    Returns:
        tuple: (a, b, f(a), f(b), new_l). With P' = {a,b} being the new population
    """
    d = lambda a, b: np.count_nonzero(a != b)

    candidates = [
        ((x1, x2, f1, f2), (max(f1, f2), min(f1, f2)), d(x1, x2), False),
        ((x1, x3, f1, f3), (max(f1, f3), min(f1, f3)), d(x1, x3), True),
        ((x2, x3, f2, f3), (max(f2, f3), min(f2, f3)), d(x2, x3), True),
    ]

    best_score = max(score for _, score, _, _ in candidates)
    best = [
        (tpl, ham, uses) for tpl, score, ham, uses in candidates if score == best_score
    ]

    if len(best) > 1:
        max_ham = max(ham for _, ham, _ in best)
        best = [item for item in best if item[1] == max_ham]

    (xa, xb, fa, fb), _, used_off = random.choice(best) if len(best) > 1 else best[0]

    new_l = 0 if used_off else l
    return xa, xb, fa, fb, new_l


def select_population_dega_a(offspring, parent, other, f_off, f_parent, f_other):
    if f_off < f_parent:
        return parent, other, f_parent, f_other
    elif f_off == f_parent:
        if np.count_nonzero(offspring != other) > np.count_nonzero(parent != other):
            return offspring, other, f_off, f_other
        else:
            return parent, other, f_parent, f_other
    else:
        return offspring, other, f_off, f_other


def select_population_alter_parent(other, parent, offspring, f_other, f_parent, f_off):
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
