"""Some utilities for the whole project."""

import logging
import h5py


def load_h5_dict(fn):
    """Load a dictionary from an HDF5 file."""
    d = {}
    with h5py.File(fn, 'r') as hf:
        for k, v in hf.items():
            d[k] = v[()]
    return d

def save_h5_dict(fn, d):
    """Save a dictionary to an HDF5 file."""
    with h5py.File(fn, 'w') as hf:
        for key, item in d.items():
            hf.create_dataset(key, data=item)

def init_logger(name):
    """Initialize a logger."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(name)s: %(message)s'))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    return logger