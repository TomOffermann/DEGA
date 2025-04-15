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
        return random.choice([(other, parent, f_other, f_parent), (other, offspring, f_other, f_off)])
      else:
        return other, offspring, f_other, f_off
    else:
      return other, parent, f_other, f_parent

class DEGA_new_non_iterated:
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
    fail = 0
    fb = 0

    while cnt < max_evals:
      # Check Convergence
      if max(f_1,f_2) >= optimum:
        print("converged in", cnt, fail, fb)
        return (max(f_1, f_2), cnt)

      if random.random() < 0.5: # Mutate
        fail += 1
        (parent, other, f_parent, f_other) = (x_1, x_2, f_1, f_2) if random.random() < 0.5 else (x_2, x_1, f_2, f_1)
        offspring = np.copy(parent)

        mask = np.random.rand(self.n) < (self.chi / self.n)
        offspring[mask] = 1 - offspring[mask]

        f_off = problem(offspring)
        cnt += 1
        x_1, x_2, f_1, f_2 = select_population(other=other, parent=parent, offspring=offspring, f_off=f_off, f_other=f_other, f_parent=f_parent)
      else:
        fb += 1
        y = biased_crossover(x_1, x_2, 0.5)
        
        if f_1 > f_2:
          x_1, x_2 = x_2, x_1
          f_1, f_2 = f_2, f_1
        
        f_y = problem(y)
        cnt += 1

        if f_y > f_1:
          hamm_dist = np.count_nonzero(f_y != f_1)
          for _ in range(int(hamm_dist*log(self.n))):
            off = biased_crossover(x_1, y, 1/hamm_dist)
            f_off = problem(off)
            cnt += 1
            
            # Update when improving bit was found
            if f_off > f_1:
              x_1 = off
              f_1 = f_off
              break
    print("exceeded max iterations", max(f_1, f_2))
    return (max(f_1, f_2), cnt)
