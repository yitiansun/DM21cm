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
from dm21cm.injections.dm import DMPWaveAnnihilationInjection
from preprocessing.step_size import pwave_phot_c_sigma, pwave_elec_c_sigma



print('\n===== Command line arguments =====')

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run_name', type=str)
parser.add_argument('-i', '--run_index', type=int)
parser.add_argument('-z', '--zf', type=str, default='002') # 01 005 002 001 0005 0002
parser.add_argument('-s', '--sf', type=int, default=10)    #  2   4  10  20   40   80
parser.add_argument('-n', '--n_threads', type=int, default=32)
parser.add_argument('-d', '--box_dim', type=int, default=128)
parser.add_argument('-c', '--channel', type=str)
parser.add_argument('--homogeneous', action='store_true')
args = parser.parse_args()
print(args)



print('\n===== Injection parameters =====')

if args.channel.startswith('pwave'):
    if args.channel == 'pwave-phot':
        m_s = 10**np.array([1.5, 12])
        c_s = pwave_phot_c_sigma(m_s)
        mass_ind, inj_ind = np.unravel_index(args.run_index, (len(m_s), 2))
        primary = 'phot_delta'

    elif args.channel == 'pwave-elec':
        m_s = 10**np.array([6.5, 8.5, 10.5, 12])
        c_s = pwave_elec_c_sigma(m_s)
        mass_ind, inj_ind = np.unravel_index(args.run_index, (len(m_s), 2))
        primary = 'elec_delta'
    else:
        raise ValueError('Invalid channel')

    m_DM = m_s[mass_ind]
    inj_multiplier = inj_ind + 1 # 1 or 2
    print('mass_ind, inj_ind:', mass_ind, inj_ind)
    injection = DMPWaveAnnihilationInjection(
        primary = primary,
        m_DM = m_s[mass_ind],
        c_sigma = c_s[mass_ind] * (inj_ind + 1),
        cell_size = 2,
    )
    print('injection:', injection)

else:
    raise ValueError('Invalid channel')

# if args.channel == 'elec':
#     primary = 'elec_delta'
#     # masses = np.logspace(6.5, 12, 12)
#     # log_lifetimes = np.array([27.994, 28.591, 28.935, 29.061, 29.004, 28.801, 28.487, 28.098, 27.670, 27.238, 26.838, 26.507]) # calibrated for fisher
#     # mass_ind, inj_ind = np.unravel_index(args.run_index, (12, 2))
#     masses = np.logspace(6.5, 12, 12)
#     masses = np.sqrt(masses[:-1] * masses[1:]) # midpoints
#     log_lifetimes = np.array([28.327, 28.793, 29.023, 29.053, 28.919, 28.656, 28.300, 27.887, 27.452, 27.032, 26.662]) # calibrated for fisher
#     mass_ind, inj_ind = np.unravel_index(args.run_index, (11, 2))

# if args.channel == 'elec':
#     primary = 'elec_delta'
#     # masses = np.logspace(6.5, 12, 12)
#     # log_lifetimes = np.array([27.994, 28.591, 28.935, 29.061, 29.004, 28.801, 28.487, 28.098, 27.670, 27.238, 26.838, 26.507]) # calibrated for fisher
#     # mass_ind, inj_ind = np.unravel_index(args.run_index, (12, 2))
#     masses = np.logspace(6.5, 12, 12)
#     masses = np.sqrt(masses[:-1] * masses[1:]) # midpoints
#     log_lifetimes = np.array([28.327, 28.793, 29.023, 29.053, 28.919, 28.656, 28.300, 27.887, 27.452, 27.032, 26.662]) # calibrated for fisher
#     mass_ind, inj_ind = np.unravel_index(args.run_index, (11, 2))

# elif args.channel == 'phot':
#     primary = 'phot_delta'
#     masses = np.logspace(1.5, 12, 22)
#     log_lifetimes = np.array([
#         29.393, 29.202, 29.012, 28.821, 28.631, 28.440, 28.250, 28.059, 27.868, 27.678, 27.487,
#         27.297, 27.106, 26.916, 26.725, 26.535, 26.344, 26.153, 25.963, 25.772, 25.582, 25.391
#     ])
#     mass_ind, inj_ind = np.unravel_index(args.run_index, (22, 2))

# else:
#     raise ValueError('Invalid channel')

# m_DM = masses[mass_ind]
# inj_multiplier = inj_ind + 1 # 1 or 2
# decay_rate = inj_multiplier * 10**(-log_lifetimes[mass_ind])
# lifetime = 1 / decay_rate
# injection = DMDecayInjection(m_DM=m_DM, lifetime=lifetime)

# print('mass_ind:', mass_ind)
# print('inj_ind:', inj_ind)
# print(f'm_DM: {m_DM:.3e}')
# print(f'lifetime: {lifetime:.3e}')



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
run_subname = f'M{mass_ind}D{inj_ind}'
run_fullname = f'{run_name}_{run_subname}'
lc_filename = f'LightCone_z5.0_HIIDIM={args.box_dim}_BOXLEN={box_len}_fisher_DM_{inj_multiplier}_r54321.h5'

# save_dir = f'/n/holyscratch01/iaifi_lab/yitians/dm21cm/prod_outputs/{run_name}/Mass_{mass_ind}/'
save_dir = f'/n/holylabs/LABS/iaifi_lab/Users/yitians/dm21cm/outputs/{run_name}/Mass_{mass_ind}/LightCones/'
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

shutil.rmtree(cache_dir)