import os
import sys
import argparse
import shutil
import numpy as np

from astropy.cosmology import Planck18
import py21cmfast as p21c

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
from dm21cm.evolve import evolve



print('\n===== Command line arguments =====')

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run_name', type=str, default='bkg')
parser.add_argument('-i', '--run_index', type=int, default=-1)
parser.add_argument('-z', '--zf', type=str, default='002') # 01 005 002 001 0005 0002
parser.add_argument('-s', '--sf', type=int, default=10)    #  2   4  10  20   40   80
parser.add_argument('-n', '--n_threads', type=int, default=32)
parser.add_argument('-d', '--box_dim', type=int, default=128)
args = parser.parse_args()
print(args)



print('\n===== Astro parameters =====')

box_len = max(256, 2 * args.box_dim)

p21c.global_params.CLUMPING_FACTOR = 1.

param_names = ['F_STAR10', 'F_STAR7_MINI', 'ALPHA_STAR', 'ALPHA_STAR_MINI', 't_STAR',
               'F_ESC10', 'F_ESC7_MINI', 'ALPHA_ESC', 'L_X', 'L_X_MINI', 'NU_X_THRESH', 'A_LW']
default_param_values = [-1.25, -2.5, 0.5, 0.0, 0.5, -1.35, -1.35, -0.3, 40.5, 40.5, 500, 2.0]
param_shifts = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.001, 0.001, 0.03, 0.03]
param_dict = dict(zip(param_names, default_param_values))
astro_params = p21c.AstroParams(param_dict)

print('box_len:', box_len)
print('global_params:', p21c.global_params)
print('astro_params:', astro_params)

if args.run_index == -1:

    print('Running the fiducial')
    run_name = args.run_name
    run_subname = 'fid'
    run_fullname = f'{run_name}_{run_subname}'
    lc_filename = f'LightCone_z5.0_HIIDIM={args.box_dim}_BOXLEN={box_len}_fisher_fid_r54321.h5'

else:
    param_index, shift_index = np.unravel_index(args.run_index, (12, 2))
    shift = param_shifts[param_index] * [-1, 1][shift_index]

    param_dict[ param_names[param_index] ] *= (1 + shift)

    if param_names[param_index] == 'ALPHA_STAR_MINI':
        param_dict[ param_names[param_index] ] = [-.1, .1][shift_index]

    print('Varied Parameter:', param_names[param_index])
    print('Run Parameter Value:', param_dict[ param_names[param_index] ])
    print('Default Parameter Value:', default_param_values[param_index])

    run_name = args.run_name
    run_subname = f'P{param_index}S{shift_index}'
    run_fullname = f'{run_name}_{run_subname}'
    lc_filename = f'LightCone_z5.0_HIIDIM={args.box_dim}_BOXLEN={box_len}_fisher_{param_names[param_index]}_{shift}_r54321.h5'

# save_dir = f'/n/holyscratch01/iaifi_lab/yitians/dm21cm/prod_outputs/{run_name}/Mass_{mass_ind}/'
save_dir = f'/n/holylabs/LABS/iaifi_lab/Users/yitians/dm21cm/outputs/{run_name}/{run_subname}/LightCones/'
os.makedirs(save_dir, exist_ok=True)

cache_dir = os.path.join(os.environ['P21C_CACHE_DIR'], run_fullname)
p21c.config['direc'] = cache_dir

print('run_name:', run_name)
print('run_subname:', run_subname)
print('run_fullname:', run_fullname)
print('lc_filename:', lc_filename)
print('save_dir:', save_dir)
print('cache_dir:', cache_dir)



print('\n===== Evolve =====')

return_dict = evolve(
    run_name = run_fullname,
    z_start = 45.,
    z_end = 5.,
    injection = None,

    p21c_initial_conditions = p21c.initial_conditions(
        user_params = p21c.UserParams(
            HII_DIM = args.box_dim,
            BOX_LEN = box_len, # [conformal Mpc]
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

return_dict['lightcone']._write(fname=lc_filename, direc=save_dir, clobber=True)

shutil.rmtree(cache_dir)