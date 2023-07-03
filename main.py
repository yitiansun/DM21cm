import os
import sys
import numpy as np
import py21cmfast as p21c

sys.path.append("..")
from dm21cm.dm_params import DMParams
from dm21cm.evolve import evolve

if __name__ == '__main__':
    
    evolve(
        run_name = 'phph_dhinit_s8zero_fine',
        run_mode = 'bath',
        z_start = 44,
        z_end = 6,
        zplusone_step_factor = 1.01,
        dm_params = DMParams(
            mode = 'swave',
            primary = 'phot_delta',
            m_DM = 1e10,
            sigmav = 1e-23,
        ),
        struct_boost_model = 'erfc 1e-3',
        enable_elec = False,
        dhinit_list = ['phot', 'T_k', 'x_e'],
        dhtf_version = '230629',
        
        p21c_initial_conditions = p21c.initial_conditions(
            user_params = p21c.UserParams(
                HII_DIM = 50, # [1] | base: 50
                BOX_LEN = 50, # [p-Mpc] | base: 50
                N_THREADS = 32
            ),
            cosmo_params = p21c.CosmoParams(
                OMm = 0.32,
                OMb = 0.049,
                POWER_INDEX = 0.96,
                SIGMA_8 = 1e-10, # base 0.83
                hlittle = 0.67
            ),
            random_seed = 54321,
            write = True,
        ),
        
        rerun_DH = False,
        clear_cache = True,
        force_reload_tf = True,
        
        use_tqdm = True,
        save_slices = False,
    )
