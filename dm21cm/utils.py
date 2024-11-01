"""Some utilities for the whole project."""

import os
import logging
import h5py


def load_h5_dict(filename):
    """Load an HDF5 file into a (nested) dictionary with auto conversion of byte strings to normal strings."""
    
    def recursive_load(group):
        """Recursive function to load a group from an HDF5 file into a nested dictionary."""
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                result[key] = recursive_load(item)
            else:
                data = item[()]
                if isinstance(data, bytes):
                    data = data.decode('utf-8')  # Convert byte strings to normal strings
                result[key] = data
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