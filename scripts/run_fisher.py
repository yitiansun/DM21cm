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

sys.path.append("../")
from scripts.step_size import StepSize250909


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--run_name', type=str, required=True)
    parser.add_argument('--log10m', type=float, help='If provided, only process this mass.')
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--dm_deriv_order', type=int, default=2, help='Order of finite derivative for DM parameter. Default 2.')
    parser.add_argument('--noise', type=str, required=True, choices=['EOS21', 'Park19'])
    args = parser.parse_args()


    #===== noise / error set =====
    # Choose the 21cmSense error set that enters the Fisher forecast.
    #   EOS21  : Errlist_SplitCore_HERA350 noise on the 23-bin EOS z-grid (default)
    #   Park19 : TotalError_HERA331 noise from Park+19 on their 12-bin z-grid
    # NB: Park19='real' makes py21cmfish re-chunk the lightcones onto the Park19
    #     z-grid and read/write PS files with a '_Park19' suffix, so the model PS
    #     must be (re)generated in that mode -- needs_new() below handles this.
    if args.noise == 'Park19':
        park19      = 'real'
        PS_suffix   = '_Park19'
        expected_nz = 12
        # Park19='real' reads noise from PS_err_dir/../21cmSense_noise_Park19/ (load_21cmsense
        # hard-codes that sibling and ignores PS_err_dir's own contents). Point straight at the
        # Park19 noise folder, as in the 21cmfish tutorial -- the '../21cmSense_noise_Park19/' in
        # the glob just resolves back to this same folder. It only exists in the repo tree.
        PS_err_dir  = py21cmfish.base_path + 'examples/data/21cmSense_noise/21cmSense_noise_Park19/'
    elif args.noise == 'EOS21':
        park19      = None
        PS_suffix   = ''
        expected_nz = 23
        PS_err_dir  = py21cmfish.base_path + 'examples/data/21cmSense_noise/21cmSense_fid_EOS21/'
    else:
        raise ValueError(f'Unknown noise set {args.noise}')

    # Generated Fisher products (PS/derivatives/global-signal .npy) are written into a
    # per-mode subfolder alongside the lightcones, so EOS21 and Park19 stay separated and
    # don't clutter the lightcone dir. Lightcones are still read from the parent dir.
    # This also gives each mode its own power_spectrum_fid_21cmsense.npy (that file is not
    # suffixed by py21cmfish), so switching modes never overwrites the other's fiducial.
    fisher_subdir = f'fisher_{args.noise}/'

    def needs_new(out_dir, param):
        """Regenerate PS on this mode's z-grid when saved products are missing or
        belong to the other mode. The fiducial PS file (power_spectrum_fid_21cmsense.npy)
        is NOT suffixed, so switching modes leaves a stale-grid file -- detect via its z-length."""
        if args.new:
            return True
        deriv_file = os.path.join(out_dir, f'power_spectrum_deriv_dict_{param}{PS_suffix}.npy')
        fid_file   = os.path.join(out_dir, 'power_spectrum_fid_21cmsense.npy')
        if not (os.path.exists(deriv_file) and os.path.exists(fid_file)):
            return True
        return np.load(fid_file, allow_pickle=True).shape[0] != expected_nz


    #===== bkg =====
    print('Processing background...')
    bkg_dir = os.environ['DM21CM_OUTPUT_DIR'] + "/bkg/"

    astro_params_vary = ['DM', 'F_STAR10', 'F_STAR7_MINI', 'ALPHA_STAR', 'ALPHA_STAR_MINI', 't_STAR',
                        'F_ESC10', 'F_ESC7_MINI', 'ALPHA_ESC', 'L_X', 'L_X_MINI', 'NU_X_THRESH', 'A_LW']
    default_param_values = [0, -1.25, -2.5, 0.5, 0.0, 0.5, -1.35, -1.35, -0.3, 40.5, 40.5, 500, 2.0]

    astro_params_fid = dict()
    for i, ap in enumerate(astro_params_vary):
        astro_params_fid[ap] = default_param_values[i]

    bkg_out = bkg_dir + fisher_subdir
    os.makedirs(bkg_out, exist_ok=True)

    params_EoS = {}
    for param in astro_params_vary[1:]:
        params_EoS[param] = py21cmfish.Parameter(
            HII_DIM=128, BOX_LEN=256, param=param,
            output_dir = bkg_out,
            lightcone_dir = bkg_dir,
            PS_err_dir = PS_err_dir,
            Park19 = park19,
            new = needs_new(bkg_out, param),
            dm_deriv_order = args.dm_deriv_order,
    )

    if args.run_name == 'bkg':
        print('Only background requested, exiting.')
        sys.exit(0)


    #===== prep =====
    print('Copying fiducial lightcone...')
    run_name = args.run_name
    channel = run_name.rsplit('-', 1)[0]
    inj_dir = os.environ['DM21CM_OUTPUT_DIR'] + f"/active/{run_name}"
    print(os.listdir(inj_dir))

    # Map each log10m sub-run to its exact on-disk folder name. Runs differ in
    # formatting (some use .3f, some .4f) but are self-consistent within a run, so we
    # reuse whatever is actually there rather than reconstructing the name.
    dir_by_log10m = {}
    for d in os.listdir(inj_dir):
        match = re.match(r'log10m([-\d\.]+)', d)
        if match:
            dir_by_log10m[float(match.group(1))] = d

    def log10m_dirname(log10m_val):
        """Exact folder name for this log10m value, matching the run's own precision."""
        for val, name in dir_by_log10m.items():
            if np.isclose(val, log10m_val):
                return name
        raise FileNotFoundError(f'No log10m directory for {log10m_val} in {inj_dir}')

    if args.log10m:
        log10m_s = np.array([args.log10m])
    else:
        log10m_s = np.sort(list(dir_by_log10m.keys()))
    m_s = 10**log10m_s
    print('Processing log10m:', log10m_s)

    EPSILON = 1e-6

    ss = StepSize250909()
    if channel == 'decay-phot':
        tau_s = ss.decay_phot_lifetime(m_s)
        inj_s = 1/tau_s
    elif channel == 'decay-elec':
        tau_s = ss.decay_elec_lifetime(m_s)
        inj_s = 1/tau_s
    elif channel.startswith('pwave-phot'):
        c_s = ss.pwave_phot_c_sigma(m_s)
        inj_s = c_s
    elif channel.startswith('pwave-elec'):
        c_s = ss.pwave_elec_c_sigma(m_s)
        inj_s = c_s
    elif channel.startswith('pwave-tau'):
        c_s = ss.pwave_tau_c_sigma(m_s)
        inj_s = c_s
    elif channel.startswith('pbhhr'):
        a_PBH = float(channel.split('-')[1][1:])
        f_s = ss.pbhhr_f(m_s, a=a_PBH)
        inj_s = f_s
    elif channel.startswith('pbhacc'):
        model = channel.split('-')[1]
        f_s = ss.pbhacc_f(m_s, model)
        inj_s = f_s

    # Copy the fiducial lightcone in each mass directory
    print('Copied :', end=' ')
    lc_filename = 'LightCone_z5.0_HIIDIM=128_BOXLEN=256_fisher_fid_r54321.h5'
    source_file = f'{bkg_dir}/{lc_filename}'
    for m in m_s:
        target_file = f'{inj_dir}/{log10m_dirname(np.log10(m))}/{lc_filename}'
        if not os.path.isfile(target_file):
            print(f'{np.log10(m):.4f}', end=' ')
            shutil.copyfile(source_file, target_file)


    #===== fisher =====
    print('Performing Fisher analysis...')
    sigma_s = []
    
    for m in tqdm(m_s):

        lc_dir = f'{inj_dir}/{log10m_dirname(np.log10(m))}/'
        lc_out = lc_dir + fisher_subdir
        os.makedirs(lc_out, exist_ok=True)

        for param in astro_params_vary[:1]:
            params_EoS[param] = py21cmfish.Parameter(
                HII_DIM=128, BOX_LEN=256, param=param,
                output_dir=lc_out,
                lightcone_dir=lc_dir,
                PS_err_dir=PS_err_dir,
                Park19=park19,
                new=needs_new(lc_out, 'DM'),
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
    print('sigma: ', sigma_s)


    #===== save =====
    print('Saving results...')
    if args.log10m:
        print('only one mass, not saving.')
    else:
        save_suffix = '' if args.noise == 'EOS21' else f'_{args.noise}'
        save_fn = os.environ['DM21CM_DIR'] + f"/outputs/limits/{run_name}{save_suffix}.txt"
        dir_path = os.path.dirname(save_fn)
        os.makedirs(dir_path, exist_ok=True)
        np.savetxt(save_fn, np.array([m_s, inj_s, sigma_s]).T, header='mass_s inj_s sigma_s')
        print('saved.')