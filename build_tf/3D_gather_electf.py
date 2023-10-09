import os
import sys
import time
from tqdm import tqdm
import argparse
import pickle

import numpy as np

if __name__ == '__main__':

    #===== Config =====
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, help='run name')
    args = parser.parse_args()
    
    run_name = args.name
    data_dir = f"{os.environ['DM21CM_DATA_DIR']}/tf/{run_name}/elec"

    abscs = load_h5_dict(f"{WDIR}/data/abscissas/abscs_{run_name}.h5")

    tfgv = np.zeros(( # rxneo. in: elec, out: phot
        len(abscs['rs']),
        len(abscs['x']),
        len(abscs['nBs']),
        len(abscs['elecEk']),
        len(abscs['photE'])
    ))
    depgv = np.zeros(( # rxneo. in: elec, out: dep_c
        len(abscs['rs']),
        len(abscs['x']),
        len(abscs['nBs']),
        len(abscs['elecEk']),
        len(abscs['dep_c'])
    )) # channels: {H ionization, He ionization, excitation, heat, continuum, xray}

    for i_nBs, nBs in enumerate(abscs['nBs']):
        tfgv_slice = np.load(f'{data_dir}/elec_tfgv_nBs{i_nBs}_rxeo.npy')
        depgv_slice = np.load(f'{data_dir}/elec_depgv_nBs{i_nBs}_rxeo.npy')
        tfgv[:, :, i_nBs, :, :] = tfgv_slice
        depgv[:, :, i_nBs, :, :] = depgv_slice
    
    #===== Save =====
    np.save(f'{data_dir}/elec_tfgv.npy', tfgv)
    np.save(f'{data_dir}/elec_depgv.npy', depgv)