import os
import sys

import numpy as np

from astropy.cosmology import Planck18
import py21cmfast as p21c

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
from dm21cm.evolve import evolve

from logo_injection import LogoInjection


if __name__ == '__main__':

    run_name = 'logo_64'

    # set global params
    p21c.global_params.CLUMPING_FACTOR = 1.
    #p21c.global_params.Pop2_ion = 0.

    box_dim = 64
    box_len = max(256, 2 * box_dim)

    return_dict = evolve(
        run_name = run_name,
        z_start = 25.,
        z_end = 5.,
        injection = LogoInjection(box_dim),
        
        p21c_initial_conditions = p21c.initial_conditions(
            user_params = p21c.UserParams(
                HII_DIM = box_dim,
                BOX_LEN = box_len, # [conformal Mpc]
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
        p21c_astro_params = p21c.AstroParams(L_X = 40.), # log10 value
        
        use_DH_init = False,
        subcycle_factor = 10,
        max_n_shell = 40,
    )

    np.save(f'{WDIR}/plotting/logo/{run_name}_records.npy', return_dict['global'])