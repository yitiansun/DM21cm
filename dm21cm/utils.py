import numpy as np

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