import os
import sys

import numpy as np
from scipy import interpolate
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo

from read import read_pbh

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
import dm21cm.physics as phys
from dm21cm.utils import load_h5_dict, save_h5_dict


def interp_dNdEdt(t_s, src_E, src_dNdEdt, tar_Ek, mass=0):
    """
    Args:
        t (1D array): time [s]
        src_E (1D array): source total energy (kinetic + mass) [eV]
        src_dNdEdt (2D array): source dNdEdt data. first dimension is time, second is energy [1/eV s]
        tar_Ek (1D array): target kinetic energy [eV]
        mass (float): mass of the particle [eV]
    """
    dNdEdt_sub = np.zeros((len(t_s), len(tar_Ek)))
    for ind in range(len(t_s)):
        interp = interpolate.interp1d(src_E, src_dNdEdt[ind], fill_value=0., bounds_error=False)
        dNdEdt_sub[ind] = interp(tar_Ek + mass)
    return t_s, dNdEdt_sub


if __name__ == '__main__':

    results_dir = '/n/home07/yitians/dm21cm/blackhawk/BlackHawk_v2.3/results'
    f_list = [f for f in os.listdir(results_dir) if f.startswith('m') and f.endswith('_pri')]
    logm_s = [float(f.split('m')[1].split('_pri')[0]) for f in f_list]
    logm_s.sort()
    print('Processing log10m = ', logm_s)
    input('(enter) > ')

    abscs = load_h5_dict(f"{os.environ['DM21CM_DATA_DIR']}/abscissas.h5")

    for logm in logm_s:

        print(f'Processing log10m = {logm:.3f}...', end='', flush=True)

        run_name = f'm{logm:.3f}_pri'
        run_dir = f"{results_dir}/{run_name}"

        evol_data = read_pbh(run_dir, 'evolution')
        # particles = output_specs(run_dir)
        # phot_pri_data = read_pbh(run_dir, 'primary', 'photon')
        # elec_pri_data = read_pbh(run_dir, 'primary', 'electron')
        phot_sec_data = read_pbh(run_dir, 'secondary', 'photon')
        elec_sec_data = read_pbh(run_dir, 'secondary', 'electron')

        # t_phot, dNdEdt_phot = interp_dNdEdt(evol_data['t'], phot_pri_data['E'], phot_pri_data['dN_dEdt'], abscs['photE'], mass=0)
        # t_elec, dNdEdt_elec = interp_dNdEdt(evol_data['t'], elec_pri_data['E'], elec_pri_data['dN_dEdt'], abscs['elecEk'], mass=phys.m_e)
        t_s, dNdEdt_phot_sec = interp_dNdEdt(evol_data['t'], phot_sec_data['E'], phot_sec_data['dN_dEdt'], abscs['photE'], mass=0)
        t_s, dNdEdt_elec_sec = interp_dNdEdt(evol_data['t'], elec_sec_data['E'], elec_sec_data['dN_dEdt'], abscs['elecEk'], mass=phys.m_e)

        # save
        data = {
            't' : t_s,
            'M' : evol_data['M'],
            'M0' : evol_data['M0'],
            'phot dNdEdt': dNdEdt_phot_sec,
            'elec dNdEdt': dNdEdt_elec_sec,
            'units' : 't: [s]; dNdEdt: [1/eV s BH]; M: [g]; M0: [g]'
        }
        save_h5_dict(f"{os.environ['DM21CM_DIR']}/data/pbh-hr/pbh_logm{logm:.3f}.h5", data)

        print('done!', flush=True)