import numpy as np

def mutate(self, individual, mutation_rate):
  mask = np.random.rand(len(individual)) < mutation_rate
  return np.where(mask, 1 - individual, individual)

def uniform_crossover(self, parent1, parent2):
  mask = np.random.rand(self.n) < 0.5
  return np.where(mask, parent1, parent2)

def biased_crossover(x1, x2, p):
  mask = np.random.rand(len(x1)) < p 
  return np.where(mask, x2, x1) 