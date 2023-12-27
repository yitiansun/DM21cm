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
    parser.add_argument('-n', '--name', type=str, default=f'{time.time():.0f}')
    args = parser.parse_args()

    run_name = f'test_{args.name}'
    ref_name = f'ref'

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
        
        p21c_initial_conditions = p21c.initial_conditions(
            user_params = p21c.UserParams(
                HII_DIM = 32,
                BOX_LEN = 256, # [conformal Mpc]
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
        
        resume = True,
        #tf_on_device = False,
        #no_injection = False,
        #use_DH_init = False,
        subcycle_factor = 10,
        max_n_shell = 20,
    )

    if args.generate:
        return_dict['lightcone']._write(fname=f'{ref_name}_lightcones.h5', direc='.', clobber=True)
    else:
        lc_ref = p21c.LightCone.read(f'{ref_name}_lightcones.h5').lightcones
        lc = return_dict['lightcone'].lightcones

        test_ks = ['Tk_box', 'x_e_box', 'brightness_temp']

        for k in test_ks:
            abs_diff = np.abs(lc[k] - lc_ref[k])
            nonzero = lc_ref[k] != 0.
            rel_diff = np.abs(lc[k] - lc_ref[k])[nonzero] / lc_ref[k][nonzero]
            print(f'{k:15}: abs_diff={np.mean(abs_diff):.6e}+/-{np.std(abs_diff):.6e}\t' + \
                          f'rel_diff={np.mean(rel_diff):.6e}+/-{np.std(rel_diff):.6e}\t')