"""Benchmarking interpolator speed."""

import os
import sys
import numpy as np
import timeit

sys.path.append("..")
from dm21cm.interpolators import BatchInterpolator as BatchInterpolatorSciPy
from dm21cm.interpolators import BatchInterpolator as BatchInterpolatorJax

rs = 20.
in_spec = np.linspace(3., 4., 500)
n_in = np.linspace(0.5, 1.5, 64**3)
x_in = np.linspace(0.01, 0.99, 64**3)
w = np.linspace(0.5, 1.5, 64**3)

bi_scipy = BatchInterpolatorSciPy(f"{os.environ['DM21CM_DATA_DIR']}/tf/230629/phot/phot_prop.h5")
bi_jax = BatchInterpolatorJax(f"{os.environ['DM21CM_DATA_DIR']}/tf/230629/phot/phot_prop.h5")

def run_bi_scipy():
    return bi_scipy(rs=rs, in_spec=in_spec, nBs_s=n_in, x_s=x_in, sum_result=True, sum_weight=w)

def run_bi_jax():
    return bi_jax(rs=rs, in_spec=in_spec, nBs_s=n_in, x_s=x_in, sum_result=True, sum_weight=w).block_until_ready()


if __name__ == '__main__':

    #elapsed_times = timeit.repeat("run_bi_scipy()", setup="from __main__ import run_bi_scipy", number=1, repeat=15)
    elapsed_times = timeit.repeat("run_bi_jax()", setup="from __main__ import run_bi_jax", number=1, repeat=30)
    print(f'{np.mean(elapsed_times[5:]):.6f} +/- {np.std(elapsed_times[5:]):.6f}')