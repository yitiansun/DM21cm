import os
import sys
import h5py
sys.path.append("..")

import logging
from dm21cm.deprecated.interpolators_jax import BatchInterpolator


# Global data structures
global_phot_dep_tf = None
#global_elec_dep_tf = None
global_phot_phot_tf = None
global_phot_prop_tf = None
global_phot_scat_tf = None
#global_elec_phot_tf = None


def load_dict(fn):
    d = {}
    with h5py.File(fn, 'r') as hf:
        for k, v in hf.items():
            d[k] = v[()]
    return d

def load_data(data_type, prefix=None, reload=False):
    """Load (global) data.
    
    Parameters:
    data_type : {'phot_dep', 'elec_dep', 'phot_phot', 'phot_prop', 'phot_scat', 'elec_phot'}
    """
    
    global global_phot_dep_tf#, global_elec_dep_tf
    global global_phot_phot_tf#, global_elec_phot_tf
    global global_phot_prop_tf, global_phot_scat_tf
    
    if prefix is None:
        prefix = os.environ['DM21CM_DATA_DIR'] + '/tf/230629/phot'
    
    if data_type == 'phot_dep':
        if (global_phot_dep_tf is None) or reload:
            global_phot_dep_tf = BatchInterpolator(f'{prefix}/phot_dep.h5')
            logging.info('Loaded photon deposition transfer function.')
        return global_phot_dep_tf
    
    # elif data_type == 'elec_dep':
    #     if (global_elec_dep_tf is None) or reload:
    #         global_elec_dep_tf = BatchInterpolator(
    #             f'{prefix}/elec_dep_renxo_aad.p'
    #         )
    #         logging.info('Loaded electron deposition transfer function.')
    #     return global_elec_dep_tf
    
    elif data_type == 'phot_prop':
        if (global_phot_prop_tf is None) or reload:
            global_phot_prop_tf = BatchInterpolator(f'{prefix}/phot_prop.h5')
            logging.info('Loaded photon propagation transfer function.')
        return global_phot_prop_tf
    
    elif data_type == 'phot_scat':
        if (global_phot_scat_tf is None) or reload:
            global_phot_scat_tf = BatchInterpolator(f'{prefix}/phot_scat.h5')
            logging.info('Loaded photon scattering transfer function.')
        return global_phot_scat_tf
    
    # elif data_type == 'elec_phot':
    #     if (global_elec_phot_tf is None) or reload:
    #         global_elec_phot_tf = BatchInterpolator(
    #             f'{prefix}/elec_phot_renxo_aad.p'
    #         )
    #         logging.info('Loaded electron photon transfer function.')
    #     return global_elec_phot_tf
    
    else:
        raise ValueError('Unknown data_type.')
