"""Some utilities for the whole project"""

import h5py

import numpy as np

#====================
# utilities

def range_wend(a, b, step=1):
    return range(int(a), int(b+1), step)

def load_dict(fn):
    d = {}
    with h5py.File(fn, 'r') as hf:
        for k, v in hf.items():
            d[k] = v[()]
    return d


#====================
# plotting

def plot_val(x):
    return np.flipud(np.log10(np.clip(np.abs(x), EPSILON, None)))

def get_circle(size):
    im = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i-(size-1)/2)**2 + (j-(size-1)/2)**2 < (0.3*size)**2:
                im[i,j] = 1
    return im

def get_circle_seq_at(size, n_per_side, at_i):
    bs = int(np.floor(size/n_per_side))
    at_i = at_i % n_per_side**2
    i = at_i // n_per_side
    j = at_i %  n_per_side
    im = np.zeros((size,size))
    im[i*bs:(i+1)*bs, j*bs:(j+1)*bs] = get_circle(bs)
    return np.einsum('i,jk->ijk', np.ones((size,)), im)