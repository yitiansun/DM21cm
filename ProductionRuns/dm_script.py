import sys, os, shutil, time
import argparse
import numpy as np

from astropy.cosmology import Planck18
import py21cmfast as p21c

is_josh = False
if is_josh:
    os.environ['DM21CM_DIR'] ='/u/jwfoster/21CM_Project/DM21cm/'
    os.environ['DH_DIR'] ='/u/jwfoster/21CM_Project/DarkHistory/'
    os.environ['DH_DATA_DIR'] ='/u/jwfoster/21CM_Project/DarkHistory/DHData/'
else:
    os.environ['DM21CM_DATA_DIR'] = '/n/holyscratch01/iaifi_lab/yitians/dm21cm/DM21cm/data/tf/zf002/data'

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)

######################################
###   Default Parameter Settings   ###
######################################

param_names = ['F_STAR10', 'F_STAR7_MINI', 'ALPHA_STAR', 'ALPHA_STAR_MINI', 't_STAR',
               'F_ESC10', 'F_ESC7_MINI', 'ALPHA_ESC', 'L_X', 'L_X_MINI', 'NU_X_THRESH', 'A_LW'
              ]
default_param_values = [-1.25, -2.5, 0.5, 0.0, 0.5, -1.35, -1.35, -0.3, 40.5, 40.5, 500, 2.0]
param_shifts = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.001, 0.001, 0.03, 0.03]

param_dict = dict(zip(param_names, default_param_values))

HII_DIM = 128
BOX_LEN = max(256, 2 * HII_DIM)

########################################################
###   Parameter Details and Command Line Arguments   ###
########################################################

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run_name', type=str)
parser.add_argument('-i', '--run_index', type=int)
parser.add_argument('-z', '--zf', type=str, default='002') # 01 005 002 001 0005 0002
parser.add_argument('-s', '--sf', type=int, default=10)    #  2   4  10  20   40   80
parser.add_argument('-c', '--channel', type=str)
parser.add_argument('-n', '--n_threads', type=int, default=32)
parser.add_argument('--homogeneous', action='store_true')
args = parser.parse_args()

if is_josh:
    os.environ['DM21CM_DATA_DIR'] = f'/u/jwfoster/21CM_Project/Data{args.zf}/'
else:
    os.environ['DM21CM_DATA_DIR'] = f'/n/holyscratch01/iaifi_lab/yitians/dm21cm/DM21cm/data/tf/zf{args.zf}/data'

print('run index:', args.run_index)
print('run name:', args.run_name)
print('n_threads:', args.n_threads)
print('zf:', args.zf)
print('sf:', args.sf)

if args.channel == 'elec':
    primary = 'elec_delta'
    masses = np.logspace(6.5, 12, 12)
    log_lifetimes = np.array([27.994, 28.591, 28.935, 29.061, 29.004, 28.801, 28.487, 28.098, 27.670, 27.238, 26.838, 26.507]) # calibrated for fisher
    mass_index, decay_index = np.unravel_index(args.run_index, (12, 2))

elif args.channel == 'phot':
    primary = 'phot_delta'
    masses = np.logspace(1.5, 12, 22)
    log_lifetimes = np.array([
        29.393, 29.202, 29.012, 28.821, 28.631, 28.440, 28.250, 28.059, 27.868, 27.678, 27.487,
        27.297, 27.106, 26.916, 26.725, 26.535, 26.344, 26.153, 25.963, 25.772, 25.582, 25.391
    ])
    mass_index, decay_index = np.unravel_index(args.run_index, (22, 2))

else:
    raise ValueError('Invalid channel')

m_DM = masses[mass_index]
decay_rate = (1 + decay_index) * 10**(-log_lifetimes[mass_index])
lifetime = 1 / decay_rate

print('DM Mass [eV]:', m_DM)
print('DM Channel:', primary)
print('DM Decay Rate [1/s]:', decay_rate)
print('DM Lifetime [s]:', lifetime)
print('Homogeneous:', args.homogeneous)

# Setting the DM Parameter in the param_dict
param_dict['DM'] = (1 + decay_index)
astro_params = p21c.AstroParams(param_dict)
print(astro_params)

#####################################
###   Setting up the Save Paths   ###
#####################################

run_name = args.run_name
run_subname = f'M{mass_index}D{1+decay_index}'
run_fullname = f'{run_name}_{run_subname}'
fname = f'LightCone_z5.0_HIIDIM={HII_DIM}_BOXLEN={BOX_LEN}_fisher_DM_{1+decay_index}_r54321.h5'

if is_josh:
    scratch_dir = '/scratch/bbta/jwfoster/21cmRuns/N'+str(HII_DIM) + '_L' + str(BOX_LEN) + '/ElecDecay/Mass' + str(mass_index) + '/'
    lightcone_direc = scratch_dir + 'LightCones/'
    cache_dir = '/tmp/' + run_subname # This is the high-performance disk for rapid i/o
    os.environ['P21C_CACHE_DIR'] = cache_dir
    p21c.config['direc'] = os.environ['P21C_CACHE_DIR']
else:
    scratch_dir = f'/n/holyscratch01/iaifi_lab/yitians/dm21cm/prod_outputs/{run_name}/Mass{mass_index}/'
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

########################################
###   Starting the Evaluation Loop   ###
########################################

# Don't waste time
if os.path.isfile(lightcone_direc + fname):
    print('Already completed')
    sys.exit()

# Only do this after all the paths have been set up. We don't want to import p21cmfast until then.
from dm21cm.dm_params import DMParams
from dm21cm.evolve import evolve

p21c.global_params.CLUMPING_FACTOR = 1.

return_dict = evolve(
    run_name = run_fullname,
    z_start = 45.,
    z_end = 5.,
    dm_params = DMParams(
        mode='decay',
        primary=primary,
        m_DM=m_DM,
        lifetime=lifetime,
    ),
    enable_elec = ('elec' in primary),

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
    no_injection = False,
    subcycle_factor = args.sf,

    homogenize_deposition = args.homogeneous,
    homogenize_injection = args.homogeneous,
)


##############################
###   Make the lightcone   ###
##############################

brightness_temp = return_dict['brightness_temp']
scrollz = return_dict['scrollz']
lightcone_quantities = ['brightness_temp','Ts_box', 'Tk_box', 'x_e_box', 'xH_box', 'density']

start = time.time()
lightcone = p21c.run_lightcone(redshift = brightness_temp.redshift,
                               user_params = brightness_temp.user_params,
                               cosmo_params = brightness_temp.cosmo_params,
                               astro_params = brightness_temp.astro_params,
                               flag_options = brightness_temp.flag_options,
                               lightcone_quantities = lightcone_quantities,
                               scrollz = scrollz,
                              )
end = time.time()
print('Time to generate lightcone:', end-start)

start = time.time()
lightcone._write(fname=fname, direc=lightcone_direc, clobber=True)
end = time.time()
print('Time to Save lightcone:', end-start)

start = time.time()
if is_josh:
    shutil.rmtree(cache_dir)
else:
    shutil.rmtree(cache_dir)
end = time.time()
print('Time to clear cache:', end-start)
sys.exit()
