import random
import numpy as np
from math import log

def biased_crossover(x1, x2, p):
  mask = np.random.rand(len(x1)) < p 
  return np.where(mask, x2, x1) 

# def select_population(x1, x2, x3, f1, f2, f3):
#   f_min = min(f1,f2)

#   if f3 < f_min:
#     return (x1, x2, f1, f2)
  
#   if f1 != f2:
#     if f3 > f_min:
#       return (x1, x3, f1, f3)
#     else:
#       if f1 == f_min:
#         d21 = np.count_nonzero(x2 != x1)
#         d23 = np.count_nonzero(x2 != x3)
#         return (x1, x2, f1, f2) if d21 > d23 else (x2, x3, f2, f3)
#       elif f2 == f_min:
#         d12 = np.count_nonzero(x1 != x2)
#         d13 = np.count_nonzero(x1 != x3)
#         return (x1, x2, f1, f2) if d12 > d13 else (x1, x3, f1, f3)
#   elif f1 == f2:
#     f12 = f1

#     d13 = np.count_nonzero(x1 != x3)
#     d23 = np.count_nonzero(x2 != x3)
#     if f3 > f12:  # x3 has better fitness
#       return (x1, x3, f12, f3) if d13 >= d23 else (x2, x3, f12, f3)

#     if f3 < f12:  # x3 has worse fitness
#       return (x1, x2, f12, f12)

#     # x3 has equal fitness
#     d12 = np.count_nonzero(x1 != x2)

#     pairs = [
#       ((x1, x2, f12, f12), d12),
#       ((x1, x3, f12, f3), d13),
#       ((x2, x3, f12, f3), d23)
#     ]

#     # Maximize Hamming distance without sorting
#     max_distance = max(d12, d13, d23)
#     candidates = [pair[0] for pair in pairs if pair[1] == max_distance]

#     # Return a random pair only if there is a tie in max distance
#     return random.choice(candidates) if len(candidates) > 1 else candidates[0]

def select_population(other, parent, offspring, f_other, f_parent, f_off, l):
    if f_off > f_parent:
      return other, offspring, f_other, f_off, 0
    elif f_off == f_parent:
      h_other_parent = np.count_nonzero(other != parent)
      h_other_offspring = np.count_nonzero(other != offspring)

      if h_other_parent > h_other_offspring:
        return other, parent, f_other, f_parent, l
      elif h_other_parent == h_other_offspring:
        return random.choice([(other, parent, f_other, f_parent, l), (other, offspring, f_other, f_off, 0)])
      else:
        return other, offspring, f_other, f_off, 0
    else:
      return other, parent, f_other, f_parent, l

class DEGA:
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

  def __call__(self, problem, optimum, max_evals, eps = 0):
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
    fail = 0
    fb = 0
    l = 0

    while cnt < max_evals:
      # Check Convergence
      if max(f_1,f_2) >= optimum:
        print("converged in", cnt, fail, fb)
        return (max(f_1, f_2), cnt)

      if f_1 != f_2 and l <= self.lmbd*log(self.n): # Exploitation Phase
        y = biased_crossover(x_1, x_2, 0.5)
        fb += 1
        if f_1 > f_2:
          x_1, x_2 = x_2, x_1
          f_1, f_2 = f_2, f_1
        
        f_y = problem(y)
        cnt += 1
        l += 1

        if f_y > f_1:
          for i in range(int(self.lmbd*log(self.n) - l)):
            off = biased_crossover(x_1, y, 1/self.lmbd)
            f_off = problem(off)
            cnt += 1
            
            # Update when improving bit was found
            if f_off > f_1:
              l = 0
              x_1 = off
              f_1 = f_off
              break
      else: # Diversity Phase
        fail += 1
        (parent, other, f_parent, f_other) = (x_1, x_2, f_1, f_2) if random.random() < 0.5 else (x_2, x_1, f_2, f_1)
        offspring = np.copy(parent)

        mask = np.random.rand(self.n) < (self.chi / self.n)
        offspring[mask] = 1 - offspring[mask]

        f_off = problem(offspring)
        cnt += 1
        x_1, x_2, f_1, f_2, l = select_population(other=other, parent=parent, offspring=offspring, f_off=f_off, f_other=f_other, f_parent=f_parent, l=l)

    print("exceeded max iterations", max(f_1, f_2))
    return (max(f_1, f_2), cnt)
