import random
import numpy as np
from matplotlib import pyplot as plt
from math import log

def leading_ones(arr):
  first_zero = np.where(arr == 0)[0]
  return first_zero[0] if len(first_zero) > 0 else len(arr)


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

class DEGA_without_uniform:
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
    H: list[tuple[float,int]] = []

    while cnt < max_evals:
      # Check Convergence
      if max(f_1,f_2) >= optimum:
        print("converged")
        return (max(f_1, f_2), cnt, H)
      
      H.append((float(np.count_nonzero(x_1 != x_2))/(self.n-min(f_1,f_2)), max(f_1, f_2)))

      if f_1 != f_2: # Exploitation Phase
        if f_1 > f_2:
          x_1, x_2 = x_2, x_1
          f_1, f_2 = f_2, f_1

        off = biased_crossover(x_1, x_2, 1/self.lmbd)
        f_off = problem(off)
        cnt += 1
        
        if f_off > f_1:
          x_1 = off
          f_1 = f_off
      else: # Diversity Phase
        (parent, other, f_parent, f_other) = (x_1, x_2, f_1, f_2) if random.random() < 0.5 else (x_2, x_1, f_2, f_1)
        offspring = np.copy(parent)

        mask = np.random.rand(self.n) < (self.chi / self.n)
        offspring[mask] = 1 - offspring[mask]

        f_off = problem(offspring)
        cnt += 1
        x_1, x_2, f_1, f_2 = select_population(x1=other, x2=parent, x3=offspring, f1=f_other, f2=f_parent, f3=f_off)

    print("exceeded max iterations", max(f_1, f_2))
    return (max(f_1, f_2), cnt, H)

def average_diversity(H_runs):
    avg_length = int(round(0.7*np.mean([len(H) for H in H_runs])))  # Compute a_h (average length)

    # Initialize an array to store averaged values up to a_h
    H_avg = np.zeros(avg_length)

    # Track how many values contribute to each index
    count_values = np.zeros(avg_length)

    # Iterate through each run and use available values up to avg_length
    for H in H_runs:
        for i in range(min(len(H), avg_length)):  # Only consider up to a_h
            H_avg[i] += H[i][0]  # Sum the values
            count_values[i] += 1  # Track the number of contributions

    # Compute element-wise average, avoiding division by zero
    H_avg = np.divide(H_avg, count_values, where=(count_values > 0))

    return H_avg

plt.rcParams["text.usetex"] = True
plt.rc('font', family='serif')
plt.figure(figsize=(12, 4))
for l, desc, color in [(10, r"$\lambda = \sqrt n$", "#D443FF"), (22, r"$\lambda = n^{2/3}$", "#7143FF"), (59, r"$\lambda = (n\log n)^{2/3}$", "#43CCFF")]:
  dega = DEGA_without_uniform(l, 100)

  H_runs = []
  for i in range(1000):
    T,C,H = dega(leading_ones, 100, (100**2)*10)
    H_runs.append(H)

  H = np.array(H)


  H_avg = average_diversity(H_runs)
  plt.plot(range(0, len(H_avg)), H_avg, color=color,label=desc)
  
plt.title(r"1000 runs averaged, $H_t/(n-f_{min})$")
plt.legend()

plt.savefig("/Users/tom/Studium/Bachelor Arbeit/Code/drop_in_h_dega.eps",bbox_inches='tight', dpi=300)
plt.show()