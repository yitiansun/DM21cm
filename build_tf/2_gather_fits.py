import numpy as np
from astropy.io import fits
import sys, os
import pickle
from tqdm import tqdm

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.spectools import get_log_bin_width

sys.path.append('..')

#---------- Config
run_name = '230408'
tf_type = 'elec'
abscs = pickle.load(open(f'../data/abscissas/abscs_{run_name}.p', 'rb'))
FITS_DIR = f'../data/tf/{run_name}/{tf_type}/ionhist_output'
SAVE_DIR = f'../data/tf/{run_name}/{tf_type}'

def fitsfn(rs, x, nBs, prefix=''):
    return f'{prefix}/tf_z_{rs:.3E}_x_{x:.3E}_nBs_{nBs:.3E}.fits'

#---------- Initialize
in_absc = abscs['photE'] if tf_type == 'phot' else abscs['elecEk']

# convention: rxneo : rs x nBs Ein out
interp_shape = (len(abscs['rs']), len(abscs['x']), len(abscs['nBs']))
EPSILON = 1e-100
hep_tf  = np.full(interp_shape + (len(in_absc), len(abscs['photE'])), EPSILON)
lep_tf  = np.full(interp_shape + (len(in_absc), len(abscs['photE'])), EPSILON)
lee_tf  = np.full(interp_shape + (len(in_absc), len(abscs['elecEk'])), EPSILON)
hed_tf  = np.full(interp_shape + (len(in_absc), len(abscs['dep_c'])-1), 0.0)
cmbloss = np.full(interp_shape + (len(in_absc),), 0.0)
lowerbound = np.full(interp_shape, 0.0)

phot_bin_width = abscs['photE'] * get_log_bin_width(abscs['photE'])
elec_bin_width = abscs['elecEk'] * get_log_bin_width(abscs['elecEk'])

pbar = tqdm(total=len(abscs['rs'])*len(abscs['x'])*len(abscs['nBs']))

#---------- Loop over
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
                
                pbar.update()
        
#---------- Save
np.save(f'{SAVE_DIR}/hep_tf_rxneo.npy', hep_tf)
np.save(f'{SAVE_DIR}/lep_tf_rxneo.npy', lep_tf)
np.save(f'{SAVE_DIR}/lee_tf_rxneo.npy', lee_tf)
np.save(f'{SAVE_DIR}/hed_tf_rxneo.npy', hed_tf)
np.save(f'{SAVE_DIR}/cmbloss_rxneo.npy', cmbloss)
np.save(f'{SAVE_DIR}/lowerbound_rxneo.npy', lowerbound)