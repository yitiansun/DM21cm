import numpy as np
from astropy.io import fits
import sys, os
import pickle
from tqdm import tqdm

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.spectools import get_log_bin_width

sys.path.append('..')

# Config
abscs = pickle.load(open('../data/abscissas/abscs_test.p', 'rb'))
FITS_DIR = '../data/idl_output/test'
SAVE_DIR = '../data/tfdata/test'

# Initialize
# convention: RNXEO
interp_shape = (len(abscs['rs']), len(abscs['x']), len(abscs['nBs']))
EPSILON = 1e-100
hep_tf  = np.full(interp_shape + (len(abscs['photE']), len(abscs['photE'])), EPSILON)
lep_tf  = np.full(interp_shape + (len(abscs['photE']), len(abscs['photE'])), EPSILON)
lee_tf  = np.full(interp_shape + (len(abscs['photE']), len(abscs['elecEk'])), EPSILON)
hed_tf  = np.full(interp_shape + (len(abscs['photE']), len(abscs['dep_c'])-1), 0.0)
cmbloss = np.full(interp_shape + (len(abscs['photE']),), 0.0)
lowerbound = np.full(interp_shape, 0.0)

def fitsfn(rs, x, nBs, prefix=''):
    return f'{prefix}/tf_z_{rs:.3E}_x_{x:.3E}_nBs_{nBs:.3E}.fits'

phot_bin_width = abscs['photE'] * get_log_bin_width(abscs['photE'])
elec_bin_width = abscs['elecEk'] * get_log_bin_width(abscs['elecEk'])

for i_rs, rs in enumerate(abscs['rs']):
    for i_x, x in enumerate(abscs['x']):
        for i_nBs, nBs in enumerate(abscs['nBs']):
            with fits.open(fitsfn(rs, x, nBs, prefix=FITS_DIR)) as f:
                hep_tf[i_rs, i_x, i_nBs] = f[1].data['hep_tf'][0] * phot_bin_width
                lep_tf[i_rs, i_x, i_nBs] = f[1].data['lep_tf'][0] * phot_bin_width
                lee_tf[i_rs, i_x, i_nBs] = f[1].data['lee_tf'][0] * elec_bin_width
                hed_tf[i_rs, i_x, i_nBs] = f[1].data['hed_tf'][0]
                cmbloss[i_rs, i_x, i_nBs] = f[1].data['cmbloss'][0]
                lowerbound[i_rs, i_x, i_nBs] = f[1].data['lowerbound'][0]
                
        
np.save(f'{SAVE_DIR}/hep_tf.npy', hep_tf)
np.save(f'{SAVE_DIR}/lep_tf.npy', lep_tf)
np.save(f'{SAVE_DIR}/lee_tf.npy', lee_tf)
np.save(f'{SAVE_DIR}/cmbloss.npy', cmbloss)
np.save(f'{SAVE_DIR}/hed_tf.npy', hed_tf)
np.save(f'{SAVE_DIR}/lowerbound.npy', lowerbound)