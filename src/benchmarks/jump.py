import numpy as np

def jump_m(arr, m):
    n = arr.size

    u = int(arr.sum())
    
    if u == n:
        return n

    if u <= n - m:
        return u

    return (n - m) - u
