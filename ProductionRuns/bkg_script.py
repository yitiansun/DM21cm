import sys, os, shutil
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
default_param_values = [-1.25, -2.5, 0.5, 0.0, 0.5, -1.35, -1.35, -0.3, 40.5, 40.5, 500, 2.0]
param_shifts = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.001, 0.001, 0.03, 0.03]

param_dict = dict(zip(param_names, default_param_values))

HII_DIM = 64
BOX_LEN = max(256, 2 * HII_DIM)

########################################################
###   Parameter Details and Command Line Arguments   ###
########################################################

run_index = int(sys.argv[1])-1
print('Run Index:', run_index)
N_THREADS = int(sys.argv[2])
print('Number of threads:', N_THREADS)

if run_index == -1:

    print('Running the fiducial')

    # Set the file and run names
    run_name = 'fid/'
    fname = 'LightCone_z5.0_HIIDIM=' + str(HII_DIM) + '_BOXLEN=' + str(BOX_LEN) + '_fisher_fid_r54321.h5'

else:
    param_index, shift_index = np.unravel_index(run_index, (12, 2))
    shift = param_shifts[param_index] * [-1, 1][shift_index]

    # Modify the value in the parameter dictionary
    param_dict[ param_names[param_index] ] *= (1 + shift)

    if param_names[param_index] == 'ALPHA_STAR_MINI':
        param_dict[ param_names[param_index] ] = [-.1, .1][shift_index]

    print('Varied Parameter:', param_names[param_index])
    print('Run Parameter Value:', param_dict[ param_names[param_index] ])
    print('Default Parameter Value:', default_param_values[param_index])

    # Set the file and run names
    run_name = 'P' + str(param_index) + 'S' + str(shift_index) + '/'
    fname = 'LightCone_z5.0_HIIDIM=' + str(HII_DIM) + '_BOXLEN=' + str(BOX_LEN) + '_fisher_' + param_names[param_index] + '_'+str(shift)+'_r54321.h5'

astro_params = p21c.AstroParams(param_dict)
print(astro_params)

# Setting up the save paths
scratch_dir = '/scratch/bbta/jwfoster/21cmRuns/N'+str(HII_DIM) + '_L' + str(BOX_LEN) + '/BKG/'
lightcone_direc = scratch_dir + 'LightCones/'
if not os.path.isdir(lightcone_direc):
    os.makedirs(lightcone_direc, exist_ok = True)

# Set up the cache dir
cache_dir = '/tmp/' + run_name # This is the high-performance disk for rapid i/o
os.environ['P21C_CACHE_DIR'] = cache_dir

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
p21c.config['direc'] = os.environ['P21C_CACHE_DIR']

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
        primary='phot_delta',
        m_DM=1e8,
        lifetime=1e50,
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
    no_injection = True,
    subcycle_factor=5,
)

##############################
###   Make the lightcone   ###
##############################

brightness_temp = return_dict['brightness_temp']
lightcone_quantities = ['brightness_temp']

lightcone = p21c.run_lightcone(redshift = brightness_temp.redshift,
                               user_params = brightness_temp.user_params,
                               cosmo_params = brightness_temp.cosmo_params,
                               astro_params = brightness_temp.astro_params,
                               flag_options=brightness_temp.flag_options,
                               lightcone_quantities=lightcone_quantities,
                              )

lightcone._write(fname=fname, direc=lightcone_direc, clobber=True)
shutil.rmtree(cache_dir)
