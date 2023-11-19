import os
import sys
import time
import argparse

import numpy as np

from astropy.cosmology import Planck18
import py21cmfast as p21c

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
from dm21cm.dm_params import DMParams
from dm21cm.evolve import evolve


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', default=False)
    args = parser.parse_args()

    generate = args.generate

    run_name = f'test_{time.time():.0f}'
    ref_name = f'test'

    os.environ['DM21CM_DATA_DIR'] = '/n/holyscratch01/iaifi_lab/yitians/dm21cm/DM21cm/data/tf/zf001/data'

    # set global params
    p21c.global_params.CLUMPING_FACTOR = 1.
    p21c.global_params.Pop2_ion = 0.

    return_dict = evolve(
        run_name = run_name,
        z_start = 45.,
        z_end = 40.,
        dm_params = DMParams(
            mode='decay',
            primary='phot_delta',
            m_DM=3e3, # [eV]
            lifetime=1e26, # [s]
        ),
        enable_elec = False,
        
        p21c_initial_conditions = p21c.initial_conditions(
            user_params = p21c.UserParams(
                HII_DIM = 32,
                BOX_LEN = 32 * 2, # [conformal Mpc]
                N_THREADS = 32,
            ),
            cosmo_params = p21c.CosmoParams(
                OMm = Planck18.Om0,
                OMb = Planck18.Ob0,
                POWER_INDEX = Planck18.meta['n'],
                SIGMA_8 = Planck18.meta['sigma8'],
                hlittle = Planck18.h,
            ),
            random_seed = 54321,
            write = True,
        ),
        p21c_astro_params = p21c.AstroParams(L_X = 0.), # log10 value
        
        clear_cache = True,
        #tf_on_device = False,
        #no_injection = False,
        #use_DH_init = False,
        subcycle_factor = 10,
        max_n_shell = 20,
    )

    if generate:
        np.save(f'{ref_name}_records.npy', return_dict['records'])
    else:
        rec_ref = np.load(f'{ref_name}_records.npy', allow_pickle=True).item()
        rec = return_dict['records']

        test_ks = ['T_k', 'x_e', 'T_k_slice', 'x_e_slice', 'x_H_slice']

        for k in test_ks:
            abs_diff = rec[k] - rec_ref[k]
            rel_diff = (rec[k] - rec_ref[k]) / rec_ref[k]
            if not isinstance(rec[k], np.ndarray):
                print(f'{k:10}: base={rec_ref[k]:.6e} ' + \
                           f'test={rec[k]:.6e}' + \
                           f'abs_diff={abs_diff:.6e}' + \
                           f'rel_diff={rel_diff:.6e}')
            else:
                print(f'{k:10}: base={np.mean(rec_ref[k]):.6e}\t\t' + \
                           f'test={np.mean(rec[k]):.6e}\t\t' + \
                           f'abs_diff={np.mean(abs_diff):.6e}+/-{np.std(abs_diff):.6e}\t\t' + \
                           f'rel_diff={np.mean(rel_diff):.6e}+/-{np.std(rel_diff):.6e}\t\t')