"""Primordial Black Hole (PBH) injection."""

import os
import sys

import numpy as np
from scipy import interpolate
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import astropy.constants as const

import jax.numpy as jnp

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.spectrum import Spectrum

sys.path.append(os.environ['DM21CM_DIR'])
from dm21cm.injections.base import Injection
import dm21cm.physics as phys


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

        self.phot_dNdEdt_table = np.load('/n/home07/yitians/dm21cm/blackhawk/analysis/phot_dNdEdt.npy') # [phot / BH eV s]
        self.elec_dNdEdt_table = np.load('/n/home07/yitians/dm21cm/blackhawk/analysis/elec_dNdEdt.npy') # [elec / BH eV s]
        self.t_arr = np.load('/n/home07/yitians/dm21cm/blackhawk/analysis/t.npy') # [s]
        # self.eng_arr = ... # [eV]
        self.phot_dNdEdt_interp = interpolate.interp1d(self.t_arr, self.phot_dNdEdt_table, axis=0) # [phot / BH s]
        self.elec_dNdEdt_interp = interpolate.interp1d(self.t_arr, self.elec_dNdEdt_table, axis=0) # [elec / BH s]

    def set_binning(self, abscs):
        # self.phot_dNdt_table = []
        # for raw_spec in self.raw_phot_dN_dEdt_per_BH:
        #     spec = Spectrum(self.eng_arr, raw_spec, spec_type='dNdE')
        #     spec.switch_spec_type('N') # [phot / BH s]
        #     spec.rebin_fast(abscs['photE'])
        #     self.phot_dNdt_table.append(spec.N)
        # self.phot_dNdt_interp = interpolate.interp1d(self.t_arr, self.phot_dNdt_table, axis=0) # [phot / BH s]

        # self.elec_dNdt_table = []
        # for raw_spec in self.raw_elec_dN_dEdt_per_BH:
        #     ind_first = np.where(self.eng_arr > phys.m_e)[0][0]
        #     spec = Spectrum(self.eng_arr[ind_first:] - phys.m_e, raw_spec[ind_first:], spec_type='dNdE')
        #     spec.switch_spec_type('N') # [elec / BH s]
        #     spec.rebin_fast(abscs['elecEk'])
        #     self.elec_dNdt_table.append(spec.N)
        # self.elec_dNdt_interp = interpolate.interp1d(self.t_arr, self.elec_dNdt_table, axis=0) # [elec / BH s]
        self.abscs = abscs

    def is_injecting_elec(self):
        return not np.allclose(self.elec_dNdEdt_table, 0.)
    
    def get_config(self):
        return {
            'mode': self.mode,
            'm_PBH': self.m_PBH,
            'f_PBH': self.f_PBH
        }
    
    #===== injections =====
    def n_PBH(self, z):
        """Mean number density of PBHs in [BH / pcm^3]."""
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