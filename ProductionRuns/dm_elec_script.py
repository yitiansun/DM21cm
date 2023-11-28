import sys, os, shutil, time
import numpy as np

from astropy.cosmology import Planck18
import py21cmfast as p21c
from py21cmfast import cache_tools
p21c.global_params.CLUMPING_FACTOR = 1.

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
default_param_values = [-1.25, -2.5, 0.5, 0.0, 0.5, -1.35, -1.35, -0.3, 40.5, 40.5, 500, 2.0]
param_shifts = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.001, 0.001, 0.03, 0.03]

param_dict = dict(zip(param_names, default_param_values))

HII_DIM = 128
BOX_LEN = max(256, 2 * HII_DIM)

########################################################
###   Parameter Details and Command Line Arguments   ###
########################################################

masses = np.geomspace(1e6, 1e12, 13)[1:]
#log_lifetimes = np.array([27., 26., 27., 28., 28., 27., 27., 26., 26., 26., 25., 25.]) + 1.
log_lifetimes = np.array([27.392, 27.989, 28.333, 28.459, 28.402, 28.199, 27.885, 27.496, 27.068, 26.636, 26.236, 25.905])

# Runtime Settings for the Dark Matter Parametrization
mass_index, decay_index = np.unravel_index(int(sys.argv[1]), (12, 2))
m_DM = masses[mass_index]
decay_rate = (1 + decay_index) * 10**(-log_lifetimes[mass_index])
lifetime = 1 / decay_rate
print('DM Mass [eV]:', m_DM)
print('DM Decay Rate [1/s]:', decay_rate)
print('DM Lifetime [s]:', lifetime)

# Setting the DM Parameter in the param_dict
param_dict['DM'] = (1 + decay_index)
astro_params = p21c.AstroParams(param_dict)
print(astro_params)

# Setting the number of threads
N_THREADS = int(sys.argv[2])

#####################################
###   Setting up the Save Paths   ###
#####################################

run_name = f'M{mass_index}D{1+decay_index}_run2/'
fname = f'LightCone_z5.0_HIIDIM={HII_DIM}_BOXLEN={BOX_LEN}_fisher_DM_{1+decay_index}_r54321.h5'

if is_josh:
    scratch_dir = '/scratch/bbta/jwfoster/21cmRuns/N'+str(HII_DIM) + '_L' + str(BOX_LEN) + '/ElecDecay/Mass' + str(mass_index) + '/'
else:
    scratch_dir = f'/n/holyscratch01/iaifi_lab/yitians/dm21cm/prod_outputs/elecdecay2/Mass{mass_index}/'
os.makedirs(scratch_dir, exist_ok=True)

lightcone_direc = scratch_dir + 'LightCones/'
os.makedirs(lightcone_direc, exist_ok=True)

# Set up the cache dir
if is_josh:
    cache_dir = '/tmp/' + run_name # This is the high-performance disk for rapid i/o
else:
    cache_dir = os.path.join(os.environ['P21C_CACHE_DIR'], run_name)

print('Run Name:', run_name)
print('Lightcone filename:', fname)
print('Saving lightcone to:', lightcone_direc)
print('Cache Dir:', cache_dir)

# Don't waste time
if os.path.isfile(lightcone_direc + fname):
    print('Already completed')
    sys.exit()

# Only do this after all the paths have been set up. We don't want to import p21cmfast until then.
from dm21cm.dm_params import DMParams
from dm21cm.evolve import evolve
if is_josh:
    os.environ['P21C_CACHE_DIR'] = cache_dir
    p21c.config['direc'] = os.environ['P21C_CACHE_DIR']
else:
    p21c.config['direc'] = cache_dir

########################################
###   Starting the Evaluation Loop   ###
########################################

p21c.global_params.CLUMPING_FACTOR = 1.

return_dict = evolve(
    run_name = run_name,
    z_start = 45.,
    z_end = 5.,
    dm_params = DMParams(
        mode='decay',
        primary='elec_delta',
        m_DM=m_DM,
        lifetime=lifetime,
    ),
    enable_elec = True,

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
    #max_n_shell = 40,
)


##############################
###   Make the lightcone   ###
##############################

brightness_temp = return_dict['brightness_temp']
scrollz = return_dict['scrollz']
lightcone_quantities = ['brightness_temp','Ts_box', 'Tk_box', 'x_e_box', 'xH_box' , 'density']

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
