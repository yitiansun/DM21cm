"""Some constants and utilities for the whole project"""

import numpy as np

# constants
EPSILON = 1e-100

Mpc = 3.08568e24 # cm

# utilities
def logspace(a, b, n):
    arr = np.logspace(np.log10(a), np.log10(b), n)
    arr[0] = a
    arr[-1] = b
    return arr

def range_wend(a, b, step=1):
    return range(int(a), int(b+1), step)

# plotting
def plot_val(x):
    return np.flipud(np.log10(np.clip(np.abs(x), EPSILON, None)))