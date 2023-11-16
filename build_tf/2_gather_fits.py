import os
import sys
from tqdm import tqdm
import argparse

import numpy as np
from astropy.io import fits

sys.path.append("..")
from dm21cm.utils import load_h5_dict

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.spectools import get_log_bin_width


if __name__ == "__main__":

    #===== Config =====
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='run name')
    parser.add_argument('-t', '--type', type=str, help="{'phot'}") # elec discontinued
    args = parser.parse_args()

    run_name = args.name
    tf_type = args.type


    #===== Initialize =====
    abscs = load_h5_dict(f"../data/abscissas/abscs_{run_name}.h5")
    fits_dir = f"{os.environ['DM21CM_DATA_DIR']}/tf/{run_name}/{tf_type}/ionhist_outputs"
    save_dir = f"{os.environ['DM21CM_DATA_DIR']}/tf/{run_name}/{tf_type}"
    in_absc = abscs['photE'] if tf_type == 'phot' else abscs['elecEk']

    # convention: rxneo : rs x nBs Ein out
    interp_shape = (len(abscs['rs']), len(abscs['x']), len(abscs['nBs']))
    EPSILON = 1e-100

    hep_tf  = np.full(interp_shape + (len(in_absc), len(abscs['photE'])), EPSILON)
    lep_tf  = np.full(interp_shape + (len(in_absc), len(abscs['photE'])), EPSILON)
    lee_tf  = np.full(interp_shape + (len(in_absc), len(abscs['elecEk'])), EPSILON)
    hed_tf  = np.full(interp_shape + (len(in_absc), 4), 0.0)
    cmbloss = np.full(interp_shape + (len(in_absc),), 0.0)
    lowerbound = np.full(interp_shape, 0.0)
    dt = np.full((len(abscs['rs']), 2), 0.0)

    phot_bin_width = abscs['photE'] * get_log_bin_width(abscs['photE'])
    elec_bin_width = abscs['elecEk'] * get_log_bin_width(abscs['elecEk'])

    pbar = tqdm(total=len(abscs['rs'])*len(abscs['x'])*len(abscs['nBs']))

    #===== Loop over =====
    for i_rs, rs in enumerate(abscs['rs']):
        for i_x, x in enumerate(abscs['x']):
            for i_nBs, nBs in enumerate(abscs['nBs']):
                fitsfn = f"{fits_dir}/{tf_type}tf_rs{i_rs}_xx{i_x}_nB{i_nBs}.fits"
                with fits.open(fitsfn) as f:
                    d = f[1].data
                    # d['name'][0][step]
                    hep_tf[i_rs, i_x, i_nBs] = np.einsum('ij,j->ij', d['hep_tf'][0][1], phot_bin_width) # N -> N
                    lep_tf[i_rs, i_x, i_nBs] = np.einsum('ij,j->ij', d['lep_tf'][0][0] + d['lep_tf'][0][1], phot_bin_width) # N -> N
                    lee_tf[i_rs, i_x, i_nBs] = np.einsum('ij,j->ij', d['lee_tf'][0][0] + d['lee_tf'][0][1], elec_bin_width) # N -> N
                    hed_tf[i_rs, i_x, i_nBs] = d['hed_tf'][0][0]*d['dt'][0][0] + d['hed_tf'][0][1]*d['dt'][0][1] # N -> E
                    cmbloss[i_rs, i_x, i_nBs] = d['cmbloss'][0][0]*d['dt'][0][0] + d['cmbloss'][0][1]*d['dt'][0][1] # N -> E
                    lowerbound[i_rs, i_x, i_nBs] = d['lowerbound'][0][1]
                    dt[i_rs] = d['dt'][0] # [step 0, step 1]
                    
                    pbar.update()
            
    #===== Save =====
    np.save(f'{save_dir}/hep_tf_rxneo.npy', hep_tf)
    np.save(f'{save_dir}/lep_tf_rxneo.npy', lep_tf)
    np.save(f'{save_dir}/lee_tf_rxneo.npy', lee_tf)
    np.save(f'{save_dir}/hed_tf_rxneo.npy', hed_tf)
    np.save(f'{save_dir}/cmbloss_rxneo.npy', cmbloss)
    np.save(f'{save_dir}/lowerbound_rxneo.npy', lowerbound)
    np.save(f'{save_dir}/dt_rxneo.npy', dt)