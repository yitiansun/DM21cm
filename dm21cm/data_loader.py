import os, sys
import logging

sys.path.append('..')

from dm21cm.interpolators import BatchInterpolator


# Global data structures
global_phot_dep_tf = None
global_elec_dep_tf = None
global_phot_phot_tf = None
global_elec_phot_tf = None


def load_data(data_type, prefix=None, reload=False):
    """Load (global) data.
    
    Parameters:
    data_type : {'phot_dep', 'elec_dep', 'phot_phot', 'elec_phot'}
    """
    
    global global_phot_dep_tf, global_elec_dep_tf
    global global_phot_phot_tf, global_elec_phot_tf
    
    if prefix is None:
        prefix = os.environ['DM21CM_DATA_DIR'] + '/tf/nBs_test_2'
    
    if data_type == 'phot_dep':
        if (global_phot_dep_tf is None) or reload:
            global_phot_dep_tf = BatchInterpolator(
                f'{prefix}/phot_dep_dlnz4.879E-2_aad.p'
            )
            logging.info('Loaded photon deposition transfer function.')
        return global_phot_dep_tf
    
    elif data_type == 'elec_dep':
        if (global_elec_dep_tf is None) or reload:
            global_elec_dep_tf = BatchInterpolator(
                f'{prefix}/elec_dep_dlnz4.879E-2_aad.p'
            )
            logging.info('Loaded electron deposition transfer function.')
        return global_elec_dep_tf
    
    elif data_type == 'phot_phot':
        if (global_phot_phot_tf is None) or reload:
            global_phot_phot_tf = BatchInterpolator(
                f'{prefix}/phot_phot_dlnz4.879E-2_aad.p'
            )
            logging.info('Loaded photon photon transfer function.')
        return global_phot_phot_tf
    
    elif data_type == 'elec_phot':
        if (global_elec_phot_tf is None) or reload:
            global_elec_phot_tf = BatchInterpolator(
                f'{prefix}/elec_phot_dlnz4.879E-2_aad.p'
            )
            logging.info('Loaded electron photon transfer function.')
        return global_elec_phot_tf
    
    else:
        raise ValueError('Unknown data_type.')