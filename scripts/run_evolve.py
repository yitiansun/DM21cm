import os
import sys
import time

import numpy as np

from astropy.cosmology import Planck18
import py21cmfast as p21c

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
from dm21cm.dm_params import DMParams
from dm21cm.evolve import evolve


if __name__ == '__main__':

    #===== config =====
    run_name = 'fc_xdecay_zf001_sf20_noxesink'
    lc_save_dir = f'/n/holyscratch01/iaifi_lab/yitians/dm21cm/outputs/fc_xdecay'
    os.makedirs(lc_save_dir, exist_ok=True)

    #--- resolution ---
    HII_DIM = 128
    BOX_LEN = max(256, 2 * HII_DIM) # [conformal Mpc]
    zf = '001'
    subcycling_factor = 20
    os.environ['DM21CM_DATA_DIR'] = f'/n/holyscratch01/iaifi_lab/yitians/dm21cm/DM21cm/data/tf/zf{zf}/data'

    #--- dark matter ---
    no_injection = False
    dm_params = DMParams(
        mode='decay',
        primary='phot_delta',
        m_DM=3e3, # [eV]
        lifetime=1e25, # [s]
    )

    #--- astrophysics ---
    p21c.global_params.CLUMPING_FACTOR = 1.
    p21c.global_params.Pop2_ion = 0.
    astro_params_option = 'noxray'

    if astro_params_option == 'prod':
        astro_params = p21c.AstroParams(
            F_STAR10 = -1.25,
            F_STAR7_MINI = -2.5,
            ALPHA_STAR = 0.5,
            ALPHA_STAR_MINI = 0.0,
            t_STAR = 0.5,
            F_ESC10 = -1.35,
            F_ESC7_MINI = -1.35,
            ALPHA_ESC = -0.3,
            L_X = 40.5,
            L_X_MINI = 40.5,
            NU_X_THRESH = 500,
            A_LW = 2.0,
        )
    elif astro_params_option == 'xray':
        astro_params = p21c.AstroParams(
            L_X = 40., # for debug runs
        )
    elif astro_params_option == 'noxray':
        astro_params = p21c.AstroParams(
            L_X = 0., # for debug runs
        )
    else:
        raise ValueError(f'Unknown astro_params_option: {astro_params_option}')

    #===== evolve =====
    return_dict = evolve(
        run_name = run_name,
        z_start = 45.,
        z_end = 5.,
        dm_params = dm_params,
        enable_elec = ('elec' in dm_params.primary),
        
        p21c_initial_conditions = p21c.initial_conditions(
            user_params = p21c.UserParams(
                HII_DIM = HII_DIM,
                BOX_LEN = BOX_LEN,
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
        p21c_astro_params = astro_params, # log10 value
        
        no_injection = no_injection,
        subcycle_factor = subcycling_factor,
        max_n_shell = 40,
    )

    np.save(f'{WDIR}/outputs/dm21cm/{run_name}_records.npy', return_dict['records'])


    #===== construct lightcone =====
    brightness_temp = return_dict['brightness_temp']
    scrollz = return_dict['scrollz']
    lightcone_quantities = ['brightness_temp','Ts_box', 'Tk_box', 'x_e_box', 'xH_box', 'density']

    timer_start = time.time()
    lightcone = p21c.run_lightcone(
        redshift = brightness_temp.redshift,
        user_params = brightness_temp.user_params,
        cosmo_params = brightness_temp.cosmo_params,
        astro_params = brightness_temp.astro_params,
        flag_options = brightness_temp.flag_options,
        lightcone_quantities = lightcone_quantities,
        scrollz = scrollz,
    )
    print(f'Time to generate lightcone: {time.time()-timer_start:.3f} s')
    lightcone._write(fname=f'lc_{run_name}.h5', direc=lc_save_dir, clobber=True)