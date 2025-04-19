import numpy as np


def leading_ones(arr):
    first_zero = np.where(arr == 0)[0]
    return first_zero[0] if len(first_zero) > 0 else len(arr)
