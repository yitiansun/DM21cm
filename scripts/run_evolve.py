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
        run_name = f'sf_xdecayx10_nodplus1_dc_noLX_noxesink_nopop2_alldepion_uddn_01atten_bath_ots_zf001',
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

        use_tqdm = True,
        #debug_flags = ['xc-bath', 'xc-custom-SFRD'],
        debug_flags = ['xc-bath', 'xc-ots', 'xc-custom-SFRD', 'xc-01attenuation'],
        #debug_flags = ['xc-ots', 'xc-custom-SFRD', 'xc-01attenuation'],
        #debug_flags = ['xc-custom-SFRD'],
        use_21cmfast_xray = False,
        debug_turn_off_pop2ion = True,
        debug_unif_delta_dep = True,
        debug_unif_delta_tf_param = True,
        st_multiplier = 10.,
        debug_nodplus1 = True,
        debug_xray_Rmax_shell = 64.,
        debug_xray_Rmax_bath = 500.,
        debug_xray_Rmax_p21c = 500.,
        debug_use_21_totinj = "sf_xdecayx10_nodplus1_noxesink_nopop2_alldepion_zf001.out",
        debug_depallion = True,
    )