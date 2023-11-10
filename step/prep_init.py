import os
import sys

from astropy.cosmology import Planck18
import py21cmfast as p21c

sys.path.append("..")
from dm21cm.dm_params import DMParams
from dm21cm.evolve import evolve

WDIR = os.environ['DM21CM_DIR']


if __name__ == '__main__':

    os.environ['DM21CM_DATA_DIR'] = '/n/holyscratch01/iaifi_lab/yitians/dm21cm/DM21cm/data/tf/zf001/data'

    # p21c.global_params.R_XLy_MAX = 500.
    # p21c.global_params.NUM_FILTER_STEPS_FOR_Ts = 40

    return_dict = evolve(
        run_name = f'stepinit_xdecay_ion_z10',
        z_start = 45.,
        z_end = 5.,
        zplusone_step_factor = 1.001,
        dm_params = DMParams(
            mode='decay',
            primary='phot_delta',
            m_DM=1e8, # [eV]
            lifetime=1e50, # [s]
        ),
        enable_elec = False,
        
        p21c_initial_conditions = p21c.initial_conditions(
            user_params = p21c.UserParams(
                HII_DIM = 32,
                BOX_LEN = 32*2, # [conformal Mpc]
                N_THREADS = 32,
                USE_INTERPOLATION_TABLES = True, # for testing
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
        
        use_21cmfast_xray = False,
        debug_turn_off_pop2ion = True,
        debug_xray_Rmax_p21c = 500.,

        debug_break_after_z = 10.,
        debug_record_extra = False,
        
        # DM21cm xray injection
        debug_flags = ['xc-ots', 'xc-custom-SFRD', 'xc-01attenuation'],
        debug_unif_delta_dep = True,
        debug_unif_delta_tf_param = True,
        st_multiplier = 10.,
        debug_nodplus1 = True,
        debug_xray_Rmax_shell = 500.,
        debug_xray_Rmax_bath = 500.,
    )