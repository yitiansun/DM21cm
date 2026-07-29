import os
import sys
import time
import argparse
import h5py
from pytest import approx

import numpy as np
from astropy.cosmology import Planck18
import py21cmfast as p21c

from dm21cm.injections.dm import DMDecayInjection
from dm21cm.evolve import evolve


def test_evolve():

    p21c.global_params.CLUMPING_FACTOR = 1.

    return_dict = evolve(
        run_name = f'test_{time.time():.0f}',
        z_start = 45.,
        z_end = 42.,
        injection = DMDecayInjection(
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
        p21c_astro_params = p21c.AstroParams(L_X = 0.),
    )

    lc = return_dict['lightcone'].lightcones
    lc_ref = p21c.LightCone.read(os.environ['DM21CM_DIR'] + '/tests/data/test_evolve_ref_lightcone.h5').lightcones

    for k in ['Tk_box', 'x_e_box', 'brightness_temp']:
        abs_diff = np.abs(lc[k] - lc_ref[k])
        nonzero = lc_ref[k] != 0.
        rel_diff = np.abs(lc[k] - lc_ref[k])[nonzero] / lc_ref[k][nonzero]
        print(f'{k:15}: abs_diff={np.mean(abs_diff):.6e}+/-{np.std(abs_diff):.6e}\t' + \
                        f'rel_diff={np.mean(rel_diff):.6e}+/-{np.std(rel_diff):.6e}\t')
        assert lc[k] == approx(lc_ref[k], rel=1e-3, abs=1e-5)



if __name__ == '__main__':

    p21c.global_params.CLUMPING_FACTOR = 1.

    return_dict = evolve(
        run_name = f'test_{time.time():.0f}',
        z_start = 45.,
        z_end = 42.,
        injection = DMDecayInjection(
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
        p21c_astro_params = p21c.AstroParams(L_X = 0.),
    )

    lc_fn = os.environ['DM21CM_DIR'] + '/tests/data/test_evolve_ref_lightcone.h5'
    return_dict['lightcone']._write(fname=lc_fn, clobber=True)
    print(f'Generated {lc_fn}')