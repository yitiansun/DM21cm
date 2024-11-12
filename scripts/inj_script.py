import os
import sys
import argparse
import numpy as np

from astropy.cosmology import Planck18
import py21cmfast as p21c

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
from dm21cm.evolve import evolve
from dm21cm.injections.pbh import PBHHRInjection
from dm21cm.injections.dm import DMDecayInjection, DMPWaveAnnihilationInjection
from preprocessing.step_size import pbh_hr_f, pwave_phot_c_sigma, pwave_elec_c_sigma



print('\n===== Command line arguments =====')

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run_name', type=str)
parser.add_argument('-i', '--run_index', type=int)
parser.add_argument('-c', '--channel', type=str)
parser.add_argument('-z', '--zf', type=str, default='002') # 01 005 002 001 0005 0002
parser.add_argument('-s', '--sf', type=int, default=10)    #  2   4  10  20   40   80
parser.add_argument('-n', '--n_threads', type=int, default=32)
parser.add_argument('-d', '--box_dim', type=int, default=128)
parser.add_argument('--homogeneous', action='store_true')
args = parser.parse_args()
print(args)



print('\n===== Injection parameters =====')

if args.channel.startswith('decay'):
    if args.channel == 'decay-test':
        injection = DMDecayInjection(
            primary='phot_delta',
            m_DM = 5e3,
            lifetime = 1e26,
        )
        inj_multiplier = 1
        mass_ind, inj_ind = 0, 0
        m_fn = 5e3

    elif args.channel == 'decay-test-2':
        injection = DMDecayInjection(
            primary='phot_delta',
            m_DM = 5e3,
            lifetime = 1e28,
        )
        inj_multiplier = 1
        mass_ind, inj_ind = 0, 0
        m_fn = 5e3

    else:
        raise ValueError('Invalid channel')

elif args.channel.startswith('pwave'):
    if args.channel == 'pwave-phot':
        #m_DM_s = 10**np.array([1.5, 4, 6, 8, 10, 12]) # [eV]
        m_DM_s = 10**np.array([2, 3, 5, 7, 9, 11]) # [eV]
        c_s = pwave_phot_c_sigma(m_DM_s) # [pcm^3/s]
        mass_ind, inj_ind = np.unravel_index(args.run_index, (len(m_DM_s), 2))
        primary = 'phot_delta'
    elif args.channel == 'pwave-elec':
        m_DM_s = 10**np.array([6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12])
        c_s = pwave_elec_c_sigma(m_DM_s)
        mass_ind, inj_ind = np.unravel_index(args.run_index, (len(m_DM_s), 2))
        primary = 'elec_delta'
    else:
        raise ValueError('Invalid channel')
    
    inj_multiplier_s = [1, 2]

    m_DM = m_DM_s[mass_ind]
    c_sigma = c_s[mass_ind]
    inj_multiplier = inj_multiplier_s[inj_ind]
    injection = DMPWaveAnnihilationInjection(
        primary = primary,
        m_DM = m_DM,
        c_sigma = c_sigma * inj_multiplier,
        cell_size = 2, # [cMpc]
    )
    m_fn = m_DM

elif args.channel == 'pbh-hr':

    # log10m_PBH_s = np.array([13.5, 15., 16.5, 18.])
    log10m_PBH_s = np.array([14, 14.5, 15.5, 16, 17, 17.5])
    inj_multiplier_s = [1, 2]

    mass_ind, inj_ind = np.unravel_index(args.run_index, (len(log10m_PBH_s), 2))
    m_PBH = 10 ** log10m_PBH_s[mass_ind] # [g]
    inj_multiplier = inj_multiplier_s[inj_ind]
    f_PBH = pbh_hr_f(m_PBH) * inj_multiplier # [1]
    injection = PBHHRInjection(
        m_PBH = m_PBH,
        f_PBH = f_PBH,
    )
    m_fn = m_PBH

else:
    raise ValueError('Invalid channel')

print('mass_ind, inj_ind:', mass_ind, inj_ind)
print('injection:', injection)



print('\n===== Save paths =====')

box_len = max(256, 2 * args.box_dim)

run_name = args.run_name
run_subname = f'log10m{np.log10(m_fn):.3f}_injm{inj_multiplier}'
run_fullname = f'{run_name}_{run_subname}'
lc_filename = f'LightCone_z5.0_HIIDIM={args.box_dim}_BOXLEN={box_len}_fisher_DM_{inj_multiplier}_r54321.h5'

save_dir = f'/n/holylabs/LABS/iaifi_lab/Users/yitians/dm21cm/outputs/{run_name}/log10m{np.log10(m_fn):.3f}/'
os.makedirs(save_dir, exist_ok=True)

cache_dir = os.path.join(os.environ['P21C_CACHE_DIR'], run_fullname)
p21c.config['direc'] = cache_dir
os.makedirs(cache_dir, exist_ok=True)

print('run_name:', run_name)
print('run_subname:', run_subname)
print('run_fullname:', run_fullname)
print('lc_filename:', lc_filename)
print('save_dir:', save_dir)
print('cache_dir:', cache_dir)



print('\n===== Default parameters =====')

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

print("cache_dir:", cache_dir)
print("cache currently not cleared")

# for entry in os.scandir(cache_dir):
#     if entry.is_file() and entry.name.endswith('.h5') and entry.name != 'lightcones.h5':
#         os.remove(entry.path)