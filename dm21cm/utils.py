"""Some utilities for the whole project."""

import os
import logging
import h5py


# def load_h5_dict(fn):
#     """Load a dictionary from an HDF5 file."""
#     d = {}
#     with h5py.File(fn, 'r') as hf:
#         for k, v in hf.items():
#             d[k] = v[()]
#     return d

# def save_h5_dict(fn, d):
#     """Save a dictionary to an HDF5 file."""
#     with h5py.File(fn, 'w') as hf:
#         for key, item in d.items():
#             hf.create_dataset(key, data=item)

def load_h5_dict(filename):
    """Load an HDF5 file into a (nested) dictionary."""
    
    def recursive_load(group):
        """Recursive function to load a group from an HDF5 file into a nested dictionary."""
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                result[key] = recursive_load(item)
            else:
                result[key] = item[()]  # Load the dataset (use [()] to retrieve the actual data)
        return result

    with h5py.File(filename, 'r') as hf:
        return recursive_load(hf)


def save_h5_dict(filename, data):
    """Save a (nested) dictionary to an HDF5 file."""

    def recursive_save(group, data):
        """Recursive function to save a nested dictionary to an HDF5 group."""
        for key, value in data.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                recursive_save(subgroup, value)
            else:
                group.create_dataset(key, data=value)

    with h5py.File(filename, 'w') as hf:
        recursive_save(hf, data)



def init_logger(name):
    """Initialize a logger."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(name)s: %(message)s'))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    return logger

abscs = load_h5_dict(f"{os.environ['DM21CM_DATA_DIR']}/abscissas.h5")