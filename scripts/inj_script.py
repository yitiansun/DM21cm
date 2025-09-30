import os
import sys
import shutil
import argparse
import numpy as np

from astropy.cosmology import Planck18
import py21cmfast as p21c

from dm21cm.evolve import evolve
from dm21cm.injections.pbh import PBHHRInjection, PBHAccretionInjection
from dm21cm.injections.dm import DMDecayInjection, DMPWaveAnnihilationInjection
from dm21cm.injections.modifiers import Multiplier

from step_size import StepSize250909 as StepSize

import darkhistory
DH_VERSION = '1.1.2.20250616'
assert darkhistory.__version__ == DH_VERSION, f'Expected darkhistory version {DH_VERSION}, got {darkhistory.__version__}.'



print('\n===== Command line arguments =====')

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run_name', type=str)
parser.add_argument('-i', '--run_index', type=int)
parser.add_argument('-c', '--channel', type=str)
parser.add_argument('-d', '--box_dim', type=int, default=128)
parser.add_argument('--n_inj_steps', type=int, default=2)
parser.add_argument('--homogeneous', action='store_true')
parser.add_argument('--save_cache', action='store_true')
parser.add_argument('--seed', type=int, default=54321)
args = parser.parse_args()
print(args)



print('\n===== Astro parameters =====')

astro_param_names = ['F_STAR10', 'F_STAR7_MINI', 'ALPHA_STAR', 'ALPHA_STAR_MINI', 't_STAR',
                     'F_ESC10', 'F_ESC7_MINI', 'ALPHA_ESC', 'L_X', 'L_X_MINI', 'NU_X_THRESH', 'A_LW']
astro_param_values = [-1.25, -2.5, 0.5, 0.0, 0.5, -1.35, -1.35, -0.3, 40.5, 40.5, 500, 2.0]
astro_param_shifts = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.001, 0.001, 0.03, 0.03]
astro_param_dict = dict(zip(astro_param_names, astro_param_values))

if args.channel == 'bkg':
    
    if args.run_index == 0: # fiducial
        print('Background fiducial')

    else:
        param_ind, shift_ind = np.unravel_index(1 + args.run_index, (12, 2))
        name = astro_param_names[param_ind]
        if name == 'ALPHA_STAR_MINI':
            astro_param_dict[name] = [-.1, .1][shift_ind]
        else:
            shift = astro_param_shifts[param_ind] * [-1, 1][shift_ind]
            astro_param_dict[name] *= (1 + shift)

        print('Background with varied parameter:', name)
        print(' - Run parameter value:', astro_param_dict[name])
        print(' - Default parameter value:', astro_param_values[param_ind])



print('\n===== Injection parameters =====')

inj_multiplier_s = np.arange(1, 1+args.n_inj_steps) # [1, 2, ..., n_inj_steps]
ss = StepSize()

if args.channel == 'bkg':
    mass_ind, inj_ind = None, None
    injection = None

elif args.channel.startswith('decay'):
    if args.channel == 'decay-phot':
        m_s = 10**np.arange(1.5, 12.01, 0.5) # [eV] | len=22
        tau_s = ss.decay_phot_lifetime(m_s)
        primary = 'phot_delta'
    elif args.channel == 'decay-elec':
        m_s = 10**np.arange(6.5, 12.01, 0.25) # [eV] | len=23
        tau_s = ss.decay_elec_lifetime(m_s)
        primary = 'elec_delta'
    else:
        raise ValueError('Invalid channel')

    mass_ind, inj_ind = np.unravel_index(args.run_index, (len(m_s), len(inj_multiplier_s)))
    m_DM = m_s[mass_ind]
    tau = tau_s[mass_ind]
    inj_multiplier = inj_multiplier_s[inj_ind]
    
    injection = DMDecayInjection(
        primary = primary,
        m_DM = m_DM,
        lifetime = tau / inj_multiplier,
        cell_size = 2, # [cMpc]
    )
    m_fn = m_DM
    astro_param_dict['DM'] = inj_multiplier

elif args.channel.startswith('pwave'): # e.g. pwave-phot[-modifier]
    
    if len(args.channel.split('-')) == 3:
        modifier = args.channel.split('-')[2]
        channel = '-'.join(args.channel.split('-')[:2])
    else:
        modifier = None
        channel = args.channel

    if channel == 'pwave-phot':
        m_s = 10**np.arange(1.5, 12.01, 0.5) # [eV] | len=22
        c_s = ss.pwave_phot_c_sigma(m_s)
        primary = 'phot_delta'
    elif channel == 'pwave-elec':
        m_s = 10**np.arange(6.5, 12.01, 0.5) # [eV] | len=12
        c_s = ss.pwave_elec_c_sigma(m_s)
        primary = 'elec_delta'
    elif channel == 'pwave-tau':
        m_s = 10**np.array([9.7, 10.0, 10.5, 11.0, 11.5, 12.0]) # [eV] | len=6
        c_s = ss.pwave_tau_c_sigma(m_s)
        primary = 'tau'
    else:
        raise ValueError('Invalid channel')

    mass_ind, inj_ind = np.unravel_index(args.run_index, (len(m_s), len(inj_multiplier_s)))
    m_DM = m_s[mass_ind]
    c_sigma = c_s[mass_ind]
    inj_multiplier = inj_multiplier_s[inj_ind]

    injection = DMPWaveAnnihilationInjection(
        primary = primary,
        m_DM = m_DM,
        c_sigma = c_sigma * inj_multiplier,
        cell_size = 2, # [cMpc]
        modifier = modifier,
    )
    m_fn = m_DM
    astro_param_dict['DM'] = inj_multiplier

elif args.channel.startswith('pbhhr'): # e.g. pbhhr-a0.999

    m_s = 10**np.arange(13.25, 18.01, 0.25) # [Msun] | len=20
    # m_s = 10**np.array([14.0625, 14.1875, 14.3125, 14.4375, 14.625])
    a_PBH = float(args.channel.split('-')[1][1:])

    mass_ind, inj_ind = np.unravel_index(args.run_index, (len(m_s), len(inj_multiplier_s)))
    m_PBH = m_s[mass_ind] # [g]
    f_PBH = ss.pbhhr_f(m_PBH, a=a_PBH) # [1]
    inj_multiplier = inj_multiplier_s[inj_ind]
    
    injection = PBHHRInjection(
        m_PBH = m_PBH,
        f_PBH = f_PBH * inj_multiplier,
        a_PBH = a_PBH,
    )
    m_fn = m_PBH
    astro_param_dict['DM'] = inj_multiplier

elif args.channel.startswith('pbhacc'): # e.g. pbhacc-PRc23

    model = args.channel.split('-')[1]
    m_s = 10**np.arange(0, 4.01, 0.5) # [M_sun] | len=18

    mass_ind, inj_ind = np.unravel_index(args.run_index, (len(m_s), len(inj_multiplier_s)))
    m_PBH = m_s[mass_ind] # [M_sun]
    f_PBH = ss.pbhacc_f(m_PBH, model) # [1]
    inj_multiplier = inj_multiplier_s[inj_ind]

    injection = PBHAccretionInjection(
        model = model,
        m_PBH = m_PBH,
        f_PBH = f_PBH * inj_multiplier,
    )
    m_fn = m_PBH
    astro_param_dict['DM'] = inj_multiplier

elif args.channel == 'none':

    mass_ind, inj_ind = None, None
    injection = None
    inj_multiplier = 1
    m_fn = 1.
    astro_param_dict['DM'] = inj_multiplier

else:
    raise ValueError('Invalid channel')

print('mass_ind, inj_ind:', mass_ind, inj_ind)
print('injection:', injection)
astro_params = p21c.AstroParams(astro_param_dict)
print('astro_params:', astro_params)


print('\n===== Other parameters =====')

p21c.global_params.CLUMPING_FACTOR = 1.
print('global_params:', p21c.global_params)
box_len = max(256, 2 * args.box_dim) # [cMpc]
print('box_len:', box_len)



print('\n===== Save paths =====')

if args.channel == 'bkg':

    run_name = args.run_name
    run_subname = 'fid' if args.run_index == 0 else f'{name}_{shift}'
    run_fullname = f'{run_name}_{run_subname}'
    lc_filename = f'LightCone_z5.0_HIIDIM={args.box_dim}_BOXLEN={box_len}_fisher_{run_subname}_r{args.seed}.h5'

    save_dir = f''
    save_dir = os.environ['DM21CM_OUTPUT_DIR'] + '/bkg/'
    os.makedirs(save_dir, exist_ok=True)

else:

    run_name = args.run_name
    run_subname = f'log10m{np.log10(m_fn):.3f}_injm{inj_multiplier}'
    run_fullname = f'{run_name}_{run_subname}'
    lc_filename = f'LightCone_z5.0_HIIDIM={args.box_dim}_BOXLEN={box_len}_fisher_DM_{inj_multiplier}_r{args.seed}.h5'

    folder_name = f'log10m{np.log10(m_fn):.3f}'
    save_dir = os.environ['DM21CM_OUTPUT_DIR'] + f'/active/{run_name}/{folder_name}/'
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



print('\n===== Evolve =====')

p21c_initial_conditions = p21c.initial_conditions(
    user_params = p21c.UserParams(
        HII_DIM = args.box_dim,
        BOX_LEN = box_len, # [conformal Mpc]
        N_THREADS = 32,
        USE_RELATIVE_VELOCITIES = True,
    ),
    cosmo_params = p21c.CosmoParams(
        OMm = Planck18.Om0,
        OMb = Planck18.Ob0,
        POWER_INDEX = Planck18.meta['n'],
        SIGMA_8 = Planck18.meta['sigma8'],
        hlittle = Planck18.h,
    ),
    random_seed = args.seed,
    write = True,
)
if args.channel.startswith('pbhacc'):
    injection.init_vcb(p21c_initial_conditions.lowres_vcb)

return_dict = evolve(
    run_name = run_fullname,
    z_start = 45.,
    z_end = 5.,
    injection = injection,

    p21c_initial_conditions = p21c_initial_conditions,
    p21c_astro_params = astro_params,

    use_DH_init = True,
    subcycle_factor = 10,

    homogenize_deposition = args.homogeneous,
    homogenize_injection = args.homogeneous,
)

return_dict['lightcone']._write(fname=lc_filename, direc=save_dir, clobber=True)



print('\n===== Finalizing =====')

print("cache_dir:", cache_dir)
if args.save_cache:
    print("cache not removed")
else:
    shutil.rmtree(cache_dir)
    print("cache removed")

with open(os.environ['DM21CM_DIR'] + '/outputs/run_results.txt', 'a') as f:
    f.write(f'{run_fullname} completed.\n')
