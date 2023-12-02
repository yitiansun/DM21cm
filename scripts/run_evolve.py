import os
import sys
import time
import shutil
import argparse

from astropy.cosmology import Planck18
import py21cmfast as p21c

sys.path.append("..")
from dm21cm.dm_params import DMParams
from dm21cm.evolve import evolve

WDIR = os.environ['DM21CM_DIR']


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-z', '--zf', type=str) # 001 002 005 01
    parser.add_argument('-s', '--sf', type=int) # 10 5 2 1
    args = parser.parse_args()

    os.environ['DM21CM_DATA_DIR'] = f'/n/holyscratch01/iaifi_lab/yitians/dm21cm/DM21cm/data/tf/zf{args.zf}/data'

    # p21c.global_params.R_XLy_MAX = 500.
    # p21c.global_params.NUM_FILTER_STEPS_FOR_Ts = 40

    run_name = f'fc_xray_128_zf{args.zf}_sf{args.sf}'
    #run_name = f'fc_xray_128_LX'

    return_dict = evolve(
        run_name = run_name,
        z_start = 45.,
        z_end = 5.,
        dm_params = DMParams(
            mode='decay',
            primary='phot_delta',
            m_DM=1e8, # [eV]
            lifetime=1e50, # [s]
        ),
        enable_elec = False,
        
        p21c_initial_conditions = p21c.initial_conditions(
            user_params = p21c.UserParams(
                HII_DIM = 128,
                BOX_LEN = 128*2, # [conformal Mpc]
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

        clear_cache = True,
        use_tqdm = True,
        debug_break_after_z = None,
        debug_record_extra = False,
        
        # 21cmFAST xray injection
        use_21cmfast_xray = False,
        debug_turn_off_pop2ion = True,
        debug_xray_Rmax_p21c = 500.,

        # DM21cm xray injection
        debug_flags = ['xc-custom-SFRD', 'xc-01attenuation'],
        debug_unif_delta_dep = True,
        debug_unif_delta_tf_param = True,
        st_multiplier = 1.,
        debug_nodplus1 = True,
        debug_xray_Rmax_shell = 430.,
        debug_xray_Rmax_bath = 430.,
        adaptive_shell = 40,

        subcycle_factor = args.sf,
        subcycle_evolve_delta = False,
    )

    # run lightcone quantities

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

    save_dir = f'/n/holyscratch01/iaifi_lab/yitians/dm21cm/prod_outputs/fc_xray'
    os.makedirs(save_dir, exist_ok=True)

    lightcone._write(fname=f'lc_{run_name}.h5', direc=save_dir, clobber=True)