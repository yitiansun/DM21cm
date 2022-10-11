import numpy as np

EPSILON = 1e-100

def plot_val(x):
    return np.flipud(np.log10(np.clip(np.abs(x), EPSILON, None)))