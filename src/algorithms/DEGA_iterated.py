import random
import numpy as np
from math import log

def biased_crossover(x1, x2, p):
  mask = np.random.rand(len(x1)) < p 
  return np.where(mask, x2, x1) 

def select_population(x1, x2, x3, f1, f2, f3):
  f_min = min(f1,f2)

  if f3 < f_min:
    return (x1, x2, f1, f2)
  
  if f1 != f2:
    if f3 > f_min:
      return (x1, x3, f1, f3)
    else:
      if f1 == f_min:
        d21 = np.count_nonzero(x2 != x1)
        d23 = np.count_nonzero(x2 != x3)
        return (x1, x2, f1, f2) if d21 > d23 else (x2, x3, f2, f3)
      elif f2 == f_min:
        d12 = np.count_nonzero(x1 != x2)
        d13 = np.count_nonzero(x1 != x3)
        return (x1, x2, f1, f2) if d12 > d13 else (x1, x3, f1, f3)
  elif f1 == f2:
    f12 = f1

    d13 = np.count_nonzero(x1 != x3)
    d23 = np.count_nonzero(x2 != x3)
    if f3 > f12:  # x3 has better fitness
      return (x1, x3, f12, f3) if d13 >= d23 else (x2, x3, f12, f3)

    if f3 < f12:  # x3 has worse fitness
      return (x1, x2, f12, f12)

    # x3 has equal fitness
    d12 = np.count_nonzero(x1 != x2)

    pairs = [
      ((x1, x2, f12, f12), d12),
      ((x1, x3, f12, f3), d13),
      ((x2, x3, f12, f3), d23)
    ]

    # Maximize Hamming distance without sorting
    max_distance = max(d12, d13, d23)
    candidates = [pair[0] for pair in pairs if pair[1] == max_distance]

    # Return a random pair only if there is a tie in max distance
    return random.choice(candidates) if len(candidates) > 1 else candidates[0]

class DEGA_iterated:
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
    """
    Run the DEGA.

    Args:
        n (int): Length of the binary string.
        lamb (int): Lambda of the algorithm
    """
    x_1 = np.random.randint(0, 2, size=self.n)
    x_2 = np.random.randint(0, 2, size=self.n)

    f_1 = problem(x_1)
    f_2 = problem(x_2)

    cnt = 2

    while cnt < max_evals:
      # Check Convergence
      if max(f_1,f_2) >= optimum:
        print("converged")
        return (max(f_1, f_2), cnt)

      if f_1 != f_2: # Exploitation Phase
        if f_1 > f_2:
          x_1, x_2 = x_2, x_1
          f_1, f_2 = f_2, f_1

        y = np.copy(x_2)
        f_y = f_2
        for _ in range(int(log(self.n))):
          offspring = biased_crossover(x_1, y, 1/2)
          f_off = problem(offspring)
          cnt += 1
        
          if f_off > f_1:
            y = offspring
            f_y = f_off
        x_1 = y
        f_1 = f_y
      else: # Diversity Phase
        offspring = np.copy(x_1 if random.random() < 0.5 else x_2)
        mask = np.random.rand(self.n) < (self.chi / self.n)
        offspring[mask] = 1 - offspring[mask]

        f_off = problem(offspring)
        cnt += 1
        x_1, x_2, f_1, f_2 = select_population(x1=x_1, x2=x_2, x3=offspring, f1=f_1, f2=f_2, f3=f_off)

    print("exceeded max iterations", max(f_1, f_2))
    return (max(f_1, f_2), cnt)
