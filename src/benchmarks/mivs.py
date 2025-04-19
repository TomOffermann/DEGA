import numpy as np


def mivs(x):
    """
    arr: np.array should be of even length
    """
    n = len(x)
    assert n % 2 == 0

    # Term 1: Sum of all elements in x
    term_1 = np.sum(x)

    # Create a matrix to store e_{i,j}
    e_matrix = np.zeros((n, n), dtype=int)

    # Define e_{i,j} based on the conditions
    for i in range(n):
        # Condition 1: j = i + 1
        if i != n // 2 - 1 and i < n - 1:  # Exclude i = n/2 - 1
            j = i + 1
            e_matrix[i, j] = 1

        # Condition 2: j = i + n//2 + 1
        if i < n // 2 - 1:
            j = i + n // 2 + 1
            e_matrix[i, j] = 1

        # Condition 3: j = i + n//2 - 1
        if i > 0 and i < n // 2:
            j = i + n // 2 - 1
            e_matrix[i, j] = 1

    x_matrix = np.outer(x, x)
    term_2 = n * np.sum(x_matrix * e_matrix)

    return term_1 - term_2
