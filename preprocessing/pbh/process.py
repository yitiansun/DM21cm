import os
import sys

import numpy as np
from scipy import interpolate
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo

from read import read_pbh, output_specs

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
import dm21cm.physics as phys
from dm21cm.utils import load_h5_dict, save_h5_dict


def t_inds(t_s, z_start, z_end):
    t_start = cosmo.age(z_start).to(u.s).value
    ind_t_start = np.searchsorted(t_s, t_start) - 1
    t_end = cosmo.age(z_end).to(u.s).value
    ind_t_end = np.searchsorted(t_s, t_end) + 1
    if ind_t_end >= len(t_s):
        ind_t_end = len(t_s) - 1
    return ind_t_start, ind_t_end

def interp_dNdEdt(t_s, src_E, src_dNdEdt, tar_Ek, mass=0):
    """
    Args:
        t (1D array): time [s]
        src_E (1D array): source total energy (kinetic + mass) [eV]
        src_dNdEdt (2D array): source dNdEdt data. first dimension is time, second is energy [1/eV s]
        tar_Ek (1D array): target kinetic energy [eV]
        mass (float): mass of the particle [eV]
    """
    ind_t_start, ind_t_end = t_inds(t_s, 4000, 5)
    t_sub = t_s[ind_t_start:ind_t_end]
    dNdEdt_sub = np.zeros((len(t_sub), len(tar_Ek)))
    for ind in range(ind_t_start, ind_t_end):
        interp = interpolate.interp1d(src_E, src_dNdEdt[ind], fill_value=0., bounds_error=False)
        dNdEdt_sub[ind-ind_t_start] = interp(tar_Ek + mass)
    return t_sub, dNdEdt_sub


if __name__ == '__main__':

    results_dir = '/n/home07/yitians/dm21cm/blackhawk/BlackHawk_v2.2/results'
    logm_s = [float(f.split('m')[1].split('_sec')[0]) for f in os.listdir(results_dir)]
    print('Processing ', logm_s)

    abscs = load_h5_dict(f"{os.environ['DM21CM_DATA_DIR']}/abscissas.h5")

    for logm in logm_s:

        print(f'Processing logm = {logm:.3f}...', end='', flush=True)

        run_name = f'm{logm:.3f}_sec'
        run_dir = f"{results_dir}/{run_name}"

        evol_data = read_pbh(run_dir, 'evolution')
        # particles = output_specs(run_dir)
        # phot_pri_data = read_pbh(run_dir, 'primary', 'photon')
        # elec_pri_data = read_pbh(run_dir, 'primary', 'electron')
        phot_sec_data = read_pbh(run_dir, 'secondary', 'photon')
        elec_sec_data = read_pbh(run_dir, 'secondary', 'electron')

        # t_phot, dNdEdt_phot = interp_dNdEdt(evol_data['t'], phot_pri_data['E'], phot_pri_data['dN_dEdt'], abscs['photE'], mass=0)
        # t_elec, dNdEdt_elec = interp_dNdEdt(evol_data['t'], elec_pri_data['E'], elec_pri_data['dN_dEdt'], abscs['elecEk'], mass=phys.m_e)
        t_phot, dNdEdt_phot_sec = interp_dNdEdt(evol_data['t'], phot_sec_data['E'], phot_sec_data['dN_dEdt'], abscs['photE'], mass=0)
        t_elec, dNdEdt_elec_sec = interp_dNdEdt(evol_data['t'], elec_sec_data['E'], elec_sec_data['dN_dEdt'], abscs['elecEk'], mass=phys.m_e)

        # save
        data = {
            't' : t_phot,
            'phot dNdEdt': dNdEdt_phot_sec,
            'elec dNdEdt': dNdEdt_elec_sec,
            'units' : 't: [s]; dNdEdt: [1/eV s BH]'
        }
        save_h5_dict(f"{os.environ['DM21CM_DIR']}/data/pbh/pbh_logm{logm:.3f}.h5", data)

        print('Done!', flush=True)