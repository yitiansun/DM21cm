"""Utilities"""

import h5py


def fitsfn(rs, x, nBs, prefix=''):
    return f"{prefix}/tf_z_{rs:.3E}_x_{x:.3E}_nBs_{nBs:.3E}.fits"


def load_dict(fn):
    
    d = {}
    with h5py.File(fn, 'r') as hf:
        for k, v in hf.items():
            d[k] = v[()]
    return d