"""Some utilities for the whole project"""

import h5py

def load_dict(fn):
    """Load a dictionary from an HDF5 file."""
    d = {}
    with h5py.File(fn, 'r') as hf:
        for k, v in hf.items():
            d[k] = v[()]
    return d