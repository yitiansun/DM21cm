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

    return_dict = evolve(
        run_name = f'sf_xdecayx100_nodplus1_dc_noLX_nopop2_uddn_zf001_Rmax32',
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
                #SIGMA_8 = 1e-6,
                hlittle = Planck18.h,
            ),
            random_seed = 54321,
            write = True,
        ),
        
        rerun_DH = False,
        clear_cache = True,
        use_tqdm = True,
        #debug_flags = ['uniform_xray'], # homogeneous injection
        #debug_flags = ['xraycheck', 'xc-noatten'], # our xray noatten to compare with 21cmfast
        debug_flags = ['xraycheck', 'xc-custom-SFRD'], # our xray ST compare with DH
        #debug_flags = ['xraycheck', 'xc-bath', 'xc-force-bath'], # our xray ST forced to bath compare with DH
        debug_astro_params = p21c.AstroParams(L_X = 0.), # log10 value
        use_DH_init = True,
        custom_YHe = 0.245, # 0.245
        debug_turn_off_pop2ion = True,
        debug_copy_dh_init = f"{WDIR}/outputs/dh/xc_xrayST_soln.p",
        track_Tk_xe = True,
        # use_21totf=f"{WDIR}/outputs/stdout/xc_nopop2_noHe_nosp_noatten_esf.out",
        # debug_even_split_f = True,
        # tf_on_device = False,
        debug_skip_dm_injection = True,
        debug_unif_delta_dep = True,
        debug_unif_delta_tf_param = True,
        st_multiplier = 100.,
        debug_nodplus1 = True,
        debug_xray_Rmax = 32.,
    )