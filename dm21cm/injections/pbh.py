"""Primordial Black Hole (PBH) injection."""

import os
import sys

import numpy as np
from scipy import interpolate
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

import jax.numpy as jnp

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.spectrum import Spectrum

sys.path.append(os.environ['DM21CM_DIR'])
from dm21cm.injections.base import Injection
import dm21cm.physics as phys

# data/pbh
#     - m_1e+15/[spectrum].txt

# injection event def = total injection in 1 second


class PBHInjection (Injection):
    """Primordial Black Hole (PBH) injection object. See parent class for details

    Args:
        m_PBH (float): PBH mass in [g].
        f_PBH (float): PBH fraction of DM.
    """

    def __init__(self, m_PBH, f_PBH=1.):
        self.mode = 'PBH'
        self.m_PBH = m_PBH
        self.f_PBH = f_PBH
        self.inj_per_sec = 1. # [inj / s] | convention: 1 injection event per second

        self.raw_phot_dN_dEdt_per_BH = ... # [phot / BH eV s]
        self.raw_elec_dN_dEdt_per_BH = ... # [elec / BH eV s]
        self.t_arr = ... # [s]
        self.eng_arr = ... # [eV]

    def set_binning(self, abscs):
        self.phot_dNdt_table = []
        for raw_spec in self.raw_phot_dN_dEdt_per_BH:
            spec = Spectrum(self.eng_arr, raw_spec, spec_type='dNdE')
            spec.switch_spec_type('N') # [phot / BH s]
            spec.rebin_fast(abscs['photE'])
            self.phot_dNdt_table.append(spec.N)
        self.phot_dNdt_interp = interpolate.interp1d(self.t_arr, self.phot_dNdt_table, axis=0) # [phot / BH s]

        self.elec_dNdt_table = []
        for raw_spec in self.raw_elec_dN_dEdt_per_BH:
            ind_first = np.where(self.eng_arr > phys.m_e)[0][0]
            spec = Spectrum(self.eng_arr[ind_first:] - phys.m_e, raw_spec[ind_first:], spec_type='dNdE')
            spec.switch_spec_type('N') # [elec / BH s]
            spec.rebin_fast(abscs['elecEk'])
            self.elec_dNdt_table.append(spec.N)
        self.elec_dNdt_interp = interpolate.interp1d(self.t_arr, self.elec_dNdt_table, axis=0) # [elec / BH s]

    def is_injecting_elec(self):
        return not np.allclose(self.elec_dNdt_table, 0.)
    
    def get_config(self):
        return {
            'mode': self.mode,
            'm_PBH': self.m_PBH,
            'f_PBH': self.f_PBH
        }
    
    #===== injections =====
    def n_PBH(self, z):
        """Mean number density of PBHs in [BH / pcm^3]."""
        return phys.rho_DM * (1+z)**3 * self.f_PBH / self.m_PBH # [BH / pcm^3]

    def inj_rate(self, z):
        return self.n_PBH(z) * self.inj_per_sec # [inj / pcm^3 s]
    
    def inj_power(self, z):
        return ... # [eV / pcm^3 s]

    def inj_phot_spec(self, z, **kwargs):
        t = cosmo.age(z).to(u.s).value # [s]
        N_phot = self.phot_dNdt_interp(t) * self.n_PBH(z) # [phot / pcm^3 s]
        return Spectrum(self.eng_arr, N_phot, spec_type='N')
    
    def inj_elec_spec(self, z, **kwargs):
        t = cosmo.age(z).to(u.s).value # [s]
        N_elec = self.elec_dNdt_interp(t) * self.n_PBH(z) # [elec / pcm^3 s]
        return Spectrum(self.eng_arr, N_elec, spec_type='N')
    
    # below are identical to decay
    def inj_phot_spec_box(self, z, delta_plus_one_box=..., **kwargs):
        box_avg = float(jnp.mean(delta_plus_one_box)) # [1] | value should be very close to 1
        return self.inj_phot_spec(z) * box_avg, delta_plus_one_box / box_avg # [phot / pcm^3 s], [1]

    def inj_elec_spec_box(self, z, delta_plus_one_box=..., **kwargs):
        box_avg = float(jnp.mean(delta_plus_one_box))
        return self.inj_elec_spec(z) * box_avg, delta_plus_one_box / box_avg # [elec / pcm^3 s], [1]