import os
import sys
import argparse
import numpy as np

from astropy.cosmology import Planck18
import py21cmfast as p21c

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)


#===== Default parameters =====

param_names = ['F_STAR10', 'F_STAR7_MINI', 'ALPHA_STAR', 'ALPHA_STAR_MINI', 't_STAR',
               'F_ESC10', 'F_ESC7_MINI', 'ALPHA_ESC', 'L_X', 'L_X_MINI', 'NU_X_THRESH', 'A_LW']
default_param_values = [-1.25, -2.5, 0.5, 0.0, 0.5, -1.35, -1.35, -0.3, 40.5, 40.5, 500, 2.0]
param_shifts = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.001, 0.001, 0.03, 0.03]

param_dict = dict(zip(param_names, default_param_values))

HII_DIM = 128
BOX_LEN = max(256, 2 * HII_DIM)


#===== Command line arguments =====

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run_name', type=str)
parser.add_argument('-i', '--run_index', type=int)
parser.add_argument('-z', '--zf', type=str, default='002') # 01 005 002 001 0005 0002
parser.add_argument('-s', '--sf', type=int, default=10)    #  2   4  10  20   40   80
parser.add_argument('-n', '--n_threads', type=int, default=32)
parser.add_argument('--homogeneous', action='store_true')
args = parser.parse_args()

print('run name:', args.run_name)
print('run index:', args.run_index)
print('zf:', args.zf)
print('sf:', args.sf)
print('n_threads:', args.n_threads)
print('homogeneous:', args.homogeneous)

# injection
mass_ind, inj_ind = np.unravel_index(args.run_index, (4, 2))
inj_multiplier = inj_ind + 1 # 1 or 2

m_PBH_s = np.array([1e15, 1e16, 1e17, 1e18])
f_PBH_s = np.array([1e-10, 3e-7, 1e-2, 1e-0])
# f_PBH_s = np.array([1e-9, 3e-6, 1e-1, 1e+1])

m_PBH = m_PBH_s[mass_ind]
f_PBH = f_PBH_s[mass_ind] * inj_multiplier

print('m_PBH:', m_PBH)
print('f_PBH:', f_PBH)

param_dict['DM'] = inj_multiplier
astro_params = p21c.AstroParams(param_dict)
print('AstroParams:', astro_params)


#===== Set up save paths =====

run_name = args.run_name
run_subname = f'M{mass_ind}D{inj_ind}'
run_fullname = f'{run_name}_{run_subname}'
fname = f'LightCone_z5.0_HIIDIM={HII_DIM}_BOXLEN={BOX_LEN}_fisher_DM_{inj_multiplier}_r54321.h5'

scratch_dir = f'/n/holyscratch01/iaifi_lab/yitians/dm21cm/prod_outputs/{run_name}/Mass_{mass_ind}/'
lightcone_direc = scratch_dir + 'LightCones/'
cache_dir = os.path.join(os.environ['P21C_CACHE_DIR'], run_fullname)
p21c.config['direc'] = cache_dir

os.makedirs(scratch_dir, exist_ok=True)
os.makedirs(lightcone_direc, exist_ok=True)

print('Run Name:', run_name)
print('Run Subname:', run_subname)
print('Lightcone filename:', fname)
print('Saving lightcone to:', lightcone_direc)
print('Cache Dir:', cache_dir)


#===== Evolve =====

os.environ['DM21CM_DATA_DIR'] = f'/n/holyscratch01/iaifi_lab/yitians/dm21cm/DM21cm/data/tf/zf{args.zf}/data'

from dm21cm.injections.pbh import PBHInjection
from dm21cm.evolve import evolve

p21c.global_params.CLUMPING_FACTOR = 1.

return_dict = evolve(
    run_name = run_fullname,
    z_start = 45.,
    z_end = 5.,
    injection = PBHInjection(m_PBH=m_PBH, f_PBH=f_PBH),

    p21c_initial_conditions = p21c.initial_conditions(
        user_params = p21c.UserParams(
            HII_DIM = HII_DIM,
            BOX_LEN = BOX_LEN, # [conformal Mpc]
            N_THREADS = args.n_threads,
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
    p21c_astro_params = astro_params,

    use_DH_init = True,
    subcycle_factor = args.sf,

    homogenize_deposition = args.homogeneous,
    homogenize_injection = args.homogeneous,
)

return_dict['lightcone']._write(fname=fname, direc=lightcone_direc, clobber=True)

# shutil.rmtree(cache_dir)