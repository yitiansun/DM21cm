import os
import sys
import shutil
import numpy as np
from scipy import stats
from tqdm import tqdm
from IPython.display import clear_output
import re
import argparse

import py21cmfish
from py21cmfish.power_spectra import *
from py21cmfish.io import *

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)

from scripts.step_size import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_name', type=str)
    parser.add_argument('--log10m', type=float, help='If provided, only process this mass.')
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--dm_deriv_order', type=int, default=2, help='Order of finite derivative for DM parameter. Default 2.')
    args = parser.parse_args()


    #===== config =====
    noise_dir = "/n/home07/yitians/dm21cm/21cmSense_fid_EOS21/"
    outputs_dir = "/n/holystore01/LABS/iaifi_lab/Users/yitians/dm21cm/outputs"


    #===== bkg =====
    print('\n\nbkg\n\n')
    bkg_dir = f"{outputs_dir}/bkg/"

    astro_params_vary = ['DM', 'F_STAR10', 'F_STAR7_MINI', 'ALPHA_STAR', 'ALPHA_STAR_MINI', 't_STAR',
                        'F_ESC10', 'F_ESC7_MINI', 'ALPHA_ESC', 'L_X', 'L_X_MINI', 'NU_X_THRESH', 'A_LW']
    default_param_values = [0, -1.25, -2.5, 0.5, 0.0, 0.5, -1.35, -1.35, -0.3, 40.5, 40.5, 500, 2.0]

    astro_params_fid = dict()
    for i, ap in enumerate(astro_params_vary):
        astro_params_fid[ap] = default_param_values[i]

    params_EoS = {}
    for param in astro_params_vary[1:]:
        params_EoS[param] = py21cmfish.Parameter(
            HII_DIM=128, BOX_LEN=256, param=param,
            output_dir = bkg_dir,
            PS_err_dir = noise_dir,
            new = False,
            dm_deriv_order = args.dm_deriv_order,
    )


    #===== prep =====
    print('\n\nprep\n\n')
    run_name = args.run_name
    channel = run_name.rsplit('-', 1)[0]
    inj_dir = f"{outputs_dir}/active/{run_name}"
    print(os.listdir(inj_dir))

    if args.log10m:
        log10m_s = np.array([args.log10m])
    else:
        log10m_s = np.sort([
            float(re.match(r'log10m([-\d\.]+)', d).group(1))
            for d in os.listdir(inj_dir)
            if re.match(r'log10m([-\d\.]+)', d)
        ])
    m_s = 10**log10m_s
    print('Processing log10m_s', log10m_s)

    EPSILON = 1e-6

    if channel == 'decay-phot':
        tau_s = decay_phot_lifetime(m_s)
        inj_s = 1/tau_s
    elif channel == 'decay-elec':
        tau_s = decay_elec_lifetime(m_s)
        inj_s = 1/tau_s
    elif channel.startswith('pwave-phot'):
        c_s = pwave_phot_c_sigma(m_s)
        inj_s = c_s
    elif channel.startswith('pwave-elec'):
        c_s = pwave_elec_c_sigma(m_s)
        inj_s = c_s
    elif channel.startswith('pwave-tau'):
        c_s = pwave_tau_c_sigma(m_s)
        inj_s = c_s
    elif channel.startswith('pbhhr'):
        a_PBH = float(channel.split('-')[1][1:])
        f_s = pbhhr_f(m_s, a=a_PBH)
        inj_s = f_s
    elif channel.startswith('pbhacc'):
        model = channel.split('-')[1]
        f_s = pbhacc_f(m_s, model)
        inj_s = f_s

    # Copy the fiducial lightcone in each mass directory
    print('Copied :', end=' ')
    for m in m_s:
        source_file = f'{bkg_dir}/LightCone_z5.0_HIIDIM=128_BOXLEN=256_fisher_fid_r54321.h5'
        target_file = f'{inj_dir}/log10m{np.log10(m):.3f}/LightCone_z5.0_HIIDIM=128_BOXLEN=256_fisher_fid_r54321.h5'
        if not os.path.isfile(target_file):
            print(f'{np.log10(m):.3f}', end=' ')
            shutil.copyfile(source_file, target_file)


    #===== fisher =====
    print('\n\nfisher\n\n')
    sigma_s = []
    
    for m in tqdm(m_s):

        lc_dir = f'{inj_dir}/log10m{np.log10(m):.3f}/'
        new = ('lc_redshifts.npy' not in os.listdir(lc_dir)) or args.new
        
        for param in astro_params_vary[:1]:
            params_EoS[param] = py21cmfish.Parameter(
                HII_DIM=128, BOX_LEN=256, param=param,
                output_dir=lc_dir,
                PS_err_dir=noise_dir, new=new,
            )

        Fij_matrix_PS, Finv_PS = py21cmfish.make_fisher_matrix(
            params_EoS,
            fisher_params=astro_params_vary,
            hpeak=0.0, obs='PS',
            k_min=0.1, k_max=1,
            sigma_mod_frac=0.2,
            add_sigma_poisson=True
        )
        sigma_s.append(np.sqrt(Finv_PS[0, 0]))
        
    sigma_s = np.array(sigma_s)
    print('sigma', sigma_s)


    #===== save =====
    print('\n\nsave\n\n')
    save_fn = f"{WDIR}/outputs/limits/{run_name}.txt"
    dir_path = os.path.dirname(save_fn)
    os.makedirs(dir_path, exist_ok=True)
    np.savetxt(save_fn, np.array([m_s, inj_s, sigma_s]).T, header='mass_s inj_s sigma_s')
    print('saved.')