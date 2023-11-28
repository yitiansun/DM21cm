import sys, os, shutil, time
import numpy as np

from astropy.cosmology import Planck18
import py21cmfast as p21c
from py21cmfast import cache_tools
p21c.global_params.CLUMPING_FACTOR = 1.

os.environ['DM21CM_DIR'] ='/u/jwfoster/21CM_Project/DM21cm/'
os.environ['DM21CM_DATA_DIR'] = '/u/jwfoster/21CM_Project/Data002/'

os.environ['DH_DIR'] ='/u/jwfoster/21CM_Project/DarkHistory/'
os.environ['DH_DATA_DIR'] ='/u/jwfoster/21CM_Project/DarkHistory/DHData/'

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

m_DM = 5e3
decay_rate = 1e-26
lifetime = 1/decay_rate

print('DM Mass [eV]:', m_DM)
print('DM Decay Rate [1/s]:', decay_rate)
print('DM Lifetime [s]:', lifetime)

########################################################
###   Parameter Details and Command Line Arguments   ###
########################################################

run_index = int(sys.argv[1])
N_THREADS = int(sys.argv[2])

if run_index == 0:
    homogenize_injection = False
    homogenize_deposition = False
    run_name = 'IFalse_DFalse/'
    fname = 'Lightcone_IFalse_DFalse.h5'

elif run_index == 1:
    homogenize_injection = False
    homogenize_deposition = True
    run_name = 'IFalse_DTrue/'
    fname = 'Lightcone_IFalse_DTrue.h5'

elif run_index == 2:
    homogenize_injection = True
    homogenize_deposition = False
    run_name = 'ITrue_DFalse/'
    fname = 'Lightcone_ITrue_DFalse.h5'

else:
    homogenize_injection = True
    homogenize_deposition = True
    run_name = 'ITrue_DTrue/'
    fname = 'Lightcone_ITrue_DTrue.h5'

#####################################
###   Setting up the Save Paths   ###
#####################################

scratch_dir = '/scratch/bbta/jwfoster/21cmRuns/SystematicVariations_NoStars/'
if not os.path.isdir(scratch_dir):
    os.makedirs(scratch_dir, exist_ok = True)

lightcone_direc = scratch_dir + 'LightCones/'
if not os.path.isdir(lightcone_direc):
    os.makedirs(lightcone_direc, exist_ok = True)

# Set up the cache dir
cache_dir = '/tmp/' + run_name # This is the high-performance disk for rapid i/o

print('Run Name:', run_name)
print('Lightcone filename:', fname)
print('Saving lightcone to:', lightcone_direc)
print('Cache Dir:,', cache_dir)

# Don't waste time
if os.path.isfile(lightcone_direc + fname):
    print('Already completed')
    sys.exit()

# Only do this after all the paths have been set up. We don't want to import p21cmfast until then.
from dm21cm.dm_params import DMParams
from dm21cm.evolve import evolve
os.environ['P21C_CACHE_DIR'] = cache_dir
p21c.config['direc'] = os.environ['P21C_CACHE_DIR']

########################################
###   Starting the Evaluation Loop   ###
########################################

p21c.global_params.CLUMPING_FACTOR = 1.
p21c.global_params.Pop2_ion = 0.
p21c.global_params.Pop3_ion = 0.

return_dict = evolve(
    run_name = run_name,
    z_start = 45.,
    z_end = 5.,
    dm_params = DMParams(
        mode='decay',
        primary='phot_delta',
        m_DM=m_DM,
        lifetime=lifetime,
    ),
    enable_elec = False,

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
    subcycle_factor=10,
    homogenize_injection = homogenize_injection,
    homogenize_deposition = homogenize_deposition,
)


##############################
###   Make the lightcone   ###
##############################

brightness_temp = return_dict['brightness_temp']
scrollz = return_dict['scrollz']
lightcone_quantities = ['brightness_temp','Ts_box', 'Tk_box', 'x_e_box', 'xH_box' ,"density"]

start = time.time()
lightcone = p21c.run_lightcone(redshift = brightness_temp.redshift,
                               user_params = brightness_temp.user_params,
                               cosmo_params = brightness_temp.cosmo_params,
                               astro_params = brightness_temp.astro_params,
                               flag_options=brightness_temp.flag_options,
                               lightcone_quantities=lightcone_quantities,
                              )
end = time.time()
print('Time to generate lightcone:', end-start)

start = time.time()
lightcone._write(fname=fname, direc=lightcone_direc, clobber=True)
end = time.time()
print('Time to Save lightcone:', end-start)

start = time.time()
shutil.rmtree(cache_dir)
end = time.time()
print('Time to clear cache:', end-start)
sys.exit()
