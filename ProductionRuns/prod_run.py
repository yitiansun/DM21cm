import sys, os, shutil, time
import argparse
import numpy as np

from astropy.cosmology import Planck18
import py21cmfast as p21c

is_josh = False
if is_josh:
    os.environ['DM21CM_DIR'] ='/u/jwfoster/21CM_Project/DM21cm/'
    os.environ['DM21CM_DATA_DIR'] = '/u/jwfoster/21CM_Project/Data002/'
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
default_param_values = [-1.25, -2.5, 0.5, 0.0, 0.5, -1.35, -1.35, -0.3, 0., 0., 500, 2.0]
param_shifts = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.001, 0.001, 0.03, 0.03]

param_dict = dict(zip(param_names, default_param_values))
param_dict['DM'] = 1
astro_params = p21c.AstroParams(param_dict)

HII_DIM = 128
BOX_LEN = max(256, 2 * HII_DIM)

########################################################
###   Parameter Details and Command Line Arguments   ###
########################################################

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--run_index', type=int)
parser.add_argument('-n', '--n_threads', type=int, default=32)
args = parser.parse_args()

run_index = args.run_index
N_THREADS = args.n_threads

m_DM = 1e7 # [eV]
channel = 'elec_delta'
decay_rate = 1e-26 # [1/s]
lifetime = 1/decay_rate
enable_elec = True

print('DM Mass [eV]:', m_DM)
print('DM Channel:', channel)
print('DM Decay Rate [1/s]:', decay_rate)
print('DM Lifetime [s]:', lifetime)

if run_index == 0:
    homogenize_injection = False
    homogenize_deposition = False
elif run_index == 1:
    homogenize_injection = False
    homogenize_deposition = True
elif run_index == 2:
    homogenize_injection = True
    homogenize_deposition = False
elif run_index == 3:
    homogenize_injection = True
    homogenize_deposition = True
else:
    raise ValueError('Invalid run index')

run_name = 'inhom_elec_m1e7'
run_subname = f'I{str(int(homogenize_injection))}_D{str(int(homogenize_deposition))}'
run_fullname = f'{run_name}_{run_subname}'
fname = f'Lightcone_{run_subname}.h5'

###########################################
###   Setting up the Save/Cache Paths   ###
###########################################

if is_josh:
    scratch_dir = f'/scratch/bbta/jwfoster/21cmRuns/{run_name}/'
    lightcone_direc = scratch_dir + 'LightCones/'
    cache_dir = '/tmp/' + run_subname # This is the high-performance disk for rapid i/o
    os.environ['P21C_CACHE_DIR'] = cache_dir
    p21c.config['direc'] = os.environ['P21C_CACHE_DIR']
else:
    scratch_dir     = os.path.join('/n/holyscratch01/iaifi_lab/yitians/dm21cm/prod_outputs', run_name)
    lightcone_direc = os.path.join(scratch_dir, 'LightCones')
    cache_dir       = os.path.join(os.environ['P21C_CACHE_DIR'], run_fullname)
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
p21c.global_params.Pop2_ion = 0.
p21c.global_params.Pop3_ion = 0.

return_dict = evolve(
    run_name = run_fullname,
    z_start = 45.,
    z_end = 5.,
    dm_params = DMParams(
        mode='decay',
        primary=channel,
        m_DM=m_DM,
        lifetime=lifetime,
    ),
    enable_elec = enable_elec,

    p21c_initial_conditions = p21c.initial_conditions(
        user_params = p21c.UserParams(
            HII_DIM = HII_DIM,
            BOX_LEN = BOX_LEN, # [conformal Mpc]
            N_THREADS = N_THREADS,
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
    subcycle_factor = 10,
    homogenize_injection = homogenize_injection,
    homogenize_deposition = homogenize_deposition,
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
end = time.time()
print('Time to clear cache:', end-start)
sys.exit()
