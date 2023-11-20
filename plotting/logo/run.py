import os
import sys

import numpy as np

from astropy.cosmology import Planck18
import py21cmfast as p21c

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
from dm21cm.dm_params import DMParams
from evolve_logo import evolve


if __name__ == '__main__':

    run_name = 'logo'

    os.environ['DM21CM_DATA_DIR'] = '/n/holyscratch01/iaifi_lab/yitians/dm21cm/DM21cm/data/tf/zf01/data'

    ref_image = np.load("source_128.npy")
    ref_image /= np.mean(ref_image)

    # set global params
    p21c.global_params.CLUMPING_FACTOR = 1.
    #p21c.global_params.Pop2_ion = 0.

    return_dict = evolve(
        run_name = run_name,
        z_start = 25.,
        z_end = 5.,
        dm_params = DMParams(
            mode='decay',
            primary='phot_delta',
            m_DM=1e3, # [eV]
            lifetime=3e25, # [s]
        ),
        enable_elec = False,
        
        p21c_initial_conditions = p21c.initial_conditions(
            user_params = p21c.UserParams(
                HII_DIM = 128,
                BOX_LEN = 128 * 2, # [conformal Mpc]
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
        
        resume = False,
        subcycle_factor = 1,
        max_n_shell = 40,
        ref_image = ref_image,
    )

    np.save(f'{WDIR}/outputs/dm21cm/{run_name}_records.npy', return_dict['records'])