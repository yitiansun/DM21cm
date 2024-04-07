import os
import sys
import argparse
import numpy as np

from astropy.cosmology import Planck18
import py21cmfast as p21c

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
from dm21cm.evolve import evolve
from dm21cm.injections.pbh import PBHInjection



print('\n===== Command line arguments =====')

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run_name', type=str)
parser.add_argument('-i', '--run_index', type=int)
parser.add_argument('-z', '--zf', type=str, default='002') # 01 005 002 001 0005 0002
parser.add_argument('-s', '--sf', type=int, default=10)    #  2   4  10  20   40   80
parser.add_argument('-n', '--n_threads', type=int, default=32)
parser.add_argument('-d', '--box_dim', type=int, default=128)
parser.add_argument('--homogeneous', action='store_true')
args = parser.parse_args()
print(args)



print('\n===== Injection parameters =====')

log10m_PBH_s = np.array([13.25, 13.75])
m_PBH_s = 10 ** log10m_PBH_s # [g]
f_PBH_s = 10 ** (3.5 * np.log10(m_PBH_s) - 63) # [1]
mass_ind, inj_ind = np.unravel_index(args.run_index, (len(m_PBH_s), 2))
inj_multiplier = inj_ind + 1 # 1 or 2
log10m_PBH = log10m_PBH_s[mass_ind]
m_PBH = m_PBH_s[mass_ind]
f_PBH = f_PBH_s[mass_ind] * inj_multiplier
injection = PBHInjection(m_PBH=m_PBH, f_PBH=f_PBH)
print('mass_ind:', mass_ind)
print('inj_ind:', inj_ind)
print(f'm_PBH: {m_PBH:.3e}')
print(f'f_PBH: {f_PBH:.3e}')



print('\n===== Default parameters =====')

box_len = max(256, 2 * args.box_dim)

p21c.global_params.CLUMPING_FACTOR = 1.

param_names = ['F_STAR10', 'F_STAR7_MINI', 'ALPHA_STAR', 'ALPHA_STAR_MINI', 't_STAR',
               'F_ESC10', 'F_ESC7_MINI', 'ALPHA_ESC', 'L_X', 'L_X_MINI', 'NU_X_THRESH', 'A_LW']
default_param_values = [-1.25, -2.5, 0.5, 0.0, 0.5, -1.35, -1.35, -0.3, 40.5, 40.5, 500, 2.0]
param_shifts = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.001, 0.001, 0.03, 0.03]
param_dict = dict(zip(param_names, default_param_values))
param_dict['DM'] = inj_multiplier
astro_params = p21c.AstroParams(param_dict)

print('box_len:', box_len)
print('global_params:', p21c.global_params)
print('astro_params:', astro_params)



print('\n===== Save paths =====')

run_name = args.run_name
run_subname = f'log10m{log10m_PBH:.3f}_injm{inj_multiplier}'
run_fullname = f'{run_name}_{run_subname}'
lc_filename = f'LightCone_z5.0_HIIDIM={args.box_dim}_BOXLEN={box_len}_fisher_DM_{inj_multiplier}_r54321.h5'

# save_dir = f'/n/holyscratch01/iaifi_lab/yitians/dm21cm/prod_outputs/{run_name}/Mass_{mass_ind}/'
save_dir = f'/n/holylabs/LABS/iaifi_lab/Users/yitians/dm21cm/outputs/{run_name}/log10m{log10m_PBH:.3f}/'
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
    injection = injection,

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



print('\n===== Clear Cache =====')

for entry in os.scandir(cache_dir):
    if entry.is_file() and entry.name.endswith('.h5') and entry.name != 'lightcones.h5':
        os.remove(entry.path)