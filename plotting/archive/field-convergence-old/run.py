import os
import sys
import argparse

import numpy as np

from astropy.cosmology import Planck18
import py21cmfast as p21c

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
from dm21cm.dm_params import DMParams
from dm21cm.evolve import evolve


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-z', '--zf', type=str)
    parser.add_argument('-s', '--sf', type=int)
    args = parser.parse_args()

    run_name = f'fc_{args.zf}'
    zff = float('0.' + args.zf)
    print(f'zf = {args.zf}, zff = {zff}, subcycle_factor = {args.sf}')

    os.environ['DM21CM_DATA_DIR'] = f'/n/holyscratch01/iaifi_lab/yitians/dm21cm/DM21cm/data/tf/zf{args.zf}/data'

    # set global params
    p21c.global_params.CLUMPING_FACTOR = 1.
    p21c.global_params.Pop2_ion = 0.

    return_dict = evolve(
        run_name = run_name,
        z_start = 45.,
        z_end = 5.,
        dm_params = DMParams(
            mode='decay',
            primary='phot_delta',
            m_DM=3e3, # [eV]
            lifetime=1e26, # [s]
        ),
        enable_elec = False,
        
        p21c_initial_conditions = p21c.initial_conditions(
            user_params = p21c.UserParams(
                HII_DIM = 64,
                BOX_LEN = 64 * 2, # [conformal Mpc]
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
        
        resume = False,
        subcycle_factor = args.sf,
        max_n_shell = 40,
    )

    np.save(f'{WDIR}/outputs/dm21cm/{run_name}_records.npy', return_dict['records'])