import os
import sys

import numpy as np

from astropy.cosmology import Planck18
import py21cmfast as p21c

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
from dm21cm.dm_params import DMParams
from dm21cm.evolve import evolve


if __name__ == '__main__':

    run_name = 'test'

    # set global params
    p21c.global_params.CLUMPING_FACTOR = 1.
    #p21c.global_params.Pop2_ion = 0.

    return_dict = evolve(
        run_name = run_name,
        z_start = 45.,
        z_end = 35.,
        zplusone_step_factor = 1.01,
        dm_params = DMParams(
            mode='decay',
            primary='phot_delta',
            m_DM=1e8, # [eV]
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
        #debug_flags = ['uniform_xray'], # homogeneous injection
        #debug_flags = ['xraycheck', 'xc-noatten'], # our xray noatten to compare with 21cmfast
        #debug_flags = ['xraycheck'], # our xray ST compare with DH
        #debug_flags = ['xraycheck', 'xc-bath', 'xc-force-bath'], # our xray ST forced to bath compare with DH
        #tf_on_device = False,
    )

    np.save(f'{run_name}_records.npy', return_dict['records'])