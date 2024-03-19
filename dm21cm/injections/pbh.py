"""Primordial Black Hole (PBH) injection."""

import os
import sys

import numpy as np
from scipy import interpolate
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import astropy.constants as const

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.spectrum import Spectrum

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
from dm21cm.injections.base import Injection
import dm21cm.physics as phys
from dm21cm.utils import load_h5_dict

data_dir = f'{WDIR}/data/pbh'


class PBHInjection (Injection):
    """Primordial Black Hole (PBH) injection object. See parent class for details.

    Args:
        m_PBH (float): PBH mass in [g].
        f_PBH (float): PBH fraction of DM.
    """

    def __init__(self, m_PBH, f_PBH=1.):
        self.mode = 'PBH'
        self.m_PBH = m_PBH
        self.f_PBH = f_PBH
        self.inj_per_sec = 1. # [inj / s] | convention: 1 injection event per second

        self.data = load_h5_dict(f'{data_dir}/pbh_logm{np.log10(m_PBH):.3f}.h5')
        self.t_arr = self.data['t'] # [s]
        zero_spec = self.data['phot dNdEdt'][0] * 0.
        self.phot_dNdEdt_interp = interpolate.interp1d(self.t_arr, self.data['phot dNdEdt'], axis=0, bounds_error=False, fill_value=zero_spec) # [phot / eV s BH]
        self.elec_dNdEdt_interp = interpolate.interp1d(self.t_arr, self.data['elec dNdEdt'], axis=0, bounds_error=False, fill_value=zero_spec) # [elec / eV s BH]

    def set_binning(self, abscs):
        self.abscs = abscs

    def is_injecting_elec(self):
        return True
    
    def get_config(self):
        return {
            'mode': self.mode,
            'm_PBH': self.m_PBH,
            'f_PBH': self.f_PBH
        }
    
    #===== injections =====
    def n_PBH(self, z):
        """Mean number density of PBHs in [BH / pcm^3]. Constant and includes 'dead' PBHs."""
        m_eV = (self.m_PBH * u.g * const.c**2).to(u.eV).value # [eV]
        return phys.rho_DM * (1+z)**3 * self.f_PBH / m_eV # [BH / pcm^3]

    def inj_rate(self, z):
        return self.n_PBH(z) * self.inj_per_sec # [inj / pcm^3 s]
    
    def inj_power(self, z):
        return self.inj_phot_spec(z).toteng() + self.inj_elec_spec(z).toteng() # [eV / pcm^3 s]

    def inj_phot_spec(self, z, **kwargs):
        t = cosmo.age(z).to(u.s).value # [s]
        dNdEdt = self.phot_dNdEdt_interp(t) * self.n_PBH(z) # [phot / pcm^3 eV s]
        return Spectrum(self.abscs['photE'], dNdEdt, spec_type='dNdE')
    
    def inj_elec_spec(self, z, **kwargs):
        t = cosmo.age(z).to(u.s).value # [s]
        dNdEdt = self.elec_dNdEdt_interp(t) * self.n_PBH(z) # [elec / pcm^3 eV s]
        return Spectrum(self.abscs['elecEk'], dNdEdt, spec_type='dNdE')
    
    def inj_phot_spec_box(self, z, delta_plus_one_box=..., **kwargs):
        return self.inj_phot_spec(z), delta_plus_one_box # [phot / pcm^3 s], [1]

    def inj_elec_spec_box(self, z, delta_plus_one_box=..., **kwargs):
        return self.inj_elec_spec(z), delta_plus_one_box # [elec / pcm^3 s], [1]