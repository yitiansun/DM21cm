import os
import sys
import time
import argparse

import numpy as np

from astropy.cosmology import Planck18
import py21cmfast as p21c

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
from dm21cm.injections.pbh import PBHHRInjection
from dm21cm.evolve import evolve


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test PBH injection')
    parser.add_argument('--run_name', type=str, default='test_pbh', help='Run name')
    parser.add_argument('--f_PBH', type=float, default=1e-8, help='PBH fraction')
    args = parser.parse_args()

    run_name = args.run_name

    os.environ['P21C_CACHE_DIR'] = '/n/home07/yitians/21cmFAST-cache' # tmp
    os.environ['P21C_CACHE_DIR'] = '/n/holyscratch01/iaifi_lab/yitians/21cmFAST-cache'
    os.environ['DM21CM_DATA_DIR'] = '/n/holyscratch01/iaifi_lab/yitians/dm21cm/DM21cm/data/tf/zf002/data'

    # set global params
    p21c.global_params.CLUMPING_FACTOR = 1.
    #p21c.global_params.Pop2_ion = 0.

    return_dict = evolve(
        run_name = run_name,
        z_start = 45.,
        z_end = 5.,
        injection = PBHHRInjection(
            m_PBH=1e15, # [g]
            f_PBH=args.f_PBH,
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
        p21c_astro_params = p21c.AstroParams(), # log10 value
        
        resume = False,
        subcycle_factor = 10,
        max_n_shell = 10,
    )

    return_dict['lightcone']._write(fname=f'{run_name}_lightcone.h5', direc='.', clobber=True)
    print(f'Generated lightcone.')