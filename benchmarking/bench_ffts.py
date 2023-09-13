"""Benchmarking fftn speed."""

import numpy as np
import jax.numpy as jnp
import timeit

box_dim = 256

z = np.random.rand(box_dim, box_dim, box_dim)
k = np.fft.rfftn(z)
zj = jnp.array(z)
kj = jnp.array(k)

def run_np():
    return np.fft.rfftn(z)

def run_inv_np():
    return np.fft.irfftn(k)

def run_jax():
    return jnp.fft.rfftn(zj)

def run_inv_jax():
    return jnp.fft.irfftn(kj)

def show_summary(elapsed_times):
    print(f'{1000*np.mean(elapsed_times[5:]):.4f} +/- {1000*np.std(elapsed_times[5:]):.4f}')


if __name__ == '__main__':

    elapsed_times = timeit.repeat("run_np()", setup="from __main__ import run_np", number=1, repeat=50)
    show_summary(elapsed_times)
    elapsed_times = timeit.repeat("run_inv_np()", setup="from __main__ import run_inv_np", number=1, repeat=50)
    show_summary(elapsed_times)

    elapsed_times = timeit.repeat("run_jax().block_until_ready()", setup="from __main__ import run_jax", number=10, repeat=50)
    show_summary(elapsed_times)
    elapsed_times = timeit.repeat("run_inv_jax().block_until_ready()", setup="from __main__ import run_inv_jax", number=10, repeat=50)
    show_summary(elapsed_times)