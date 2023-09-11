import os
import sys
import h5py
import logging

sys.path.append("..")
#from dm21cm.deprecated.interpolators_jax import BatchInterpolator
from dm21cm.interpolators import BatchInterpolator


global_phot_dep_tf = None
global_elec_dep_tf = None
global_phot_prop_tf = None
global_phot_scat_tf = None
global_elec_scat_tf = None


def load_dict(fn):
    """Load a dictionary from an HDF5 file."""
    d = {}
    with h5py.File(fn, 'r') as hf:
        for k, v in hf.items():
            d[k] = v[()]
    return d


def load_data(data_type, prefix=None, reload=False):
    """Load (global) data.
    
    Args:
        data_type {'phot_dep', 'elec_dep', 'phot_phot', 'phot_prop', 'phot_scat', 'elec_scat'}
        prefix (str): path to data directory
        reload (bool): force reload if True
    """
    
    global global_phot_dep_tf, global_elec_dep_tf
    global global_phot_prop_tf, global_phot_scat_tf
    global global_elec_scat_tf
    
    if prefix is None:
        prefix = os.environ['DM21CM_DATA_DIR'] + '/tf/230629'
    
    if data_type == 'phot_dep':
        if (global_phot_dep_tf is None) or reload:
            global_phot_dep_tf = BatchInterpolator(f'{prefix}/phot/phot_dep.h5')
            logging.info('Loaded photon deposition transfer function.')
        return global_phot_dep_tf
    
    elif data_type == 'elec_dep':
        if (global_elec_dep_tf is None) or reload:
            global_elec_dep_tf = BatchInterpolator(f'{prefix}/elec/elec_dep.h5')
            logging.info('Loaded electron deposition transfer function.')
        return global_elec_dep_tf
    
    elif data_type == 'phot_prop':
        if (global_phot_prop_tf is None) or reload:
            global_phot_prop_tf = BatchInterpolator(f'{prefix}/phot/phot_prop.h5')
            logging.info('Loaded photon propagation transfer function.')
        return global_phot_prop_tf
    
    elif data_type == 'phot_scat':
        if (global_phot_scat_tf is None) or reload:
            global_phot_scat_tf = BatchInterpolator(f'{prefix}/phot/phot_scat.h5')
            logging.info('Loaded photon scattering transfer function.')
        return global_phot_scat_tf
    
    elif data_type == 'elec_scat':
        if (global_elec_scat_tf is None) or reload:
            global_elec_scat_tf = BatchInterpolator(f'{prefix}/elec/elec_scat.h5')
            logging.info('Loaded electron scattering transfer function.')
        return global_elec_scat_tf
    
    else:
        raise ValueError('Unknown data_type.')
