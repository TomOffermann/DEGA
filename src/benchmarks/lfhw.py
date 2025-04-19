import numpy as np


def linear_harmonic(arr):
    indices = np.arange(1, len(arr) + 1)
    return np.sum(indices * arr)
