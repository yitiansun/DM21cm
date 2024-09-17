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

        self.m_eV = (self.m_PBH * u.g * const.c**2).to(u.eV).value # [eV]

        try:
            self.data = load_h5_dict(f'{data_dir}/pbh_logm{np.log10(m_PBH):.3f}.h5')
        except FileNotFoundError:
            raise FileNotFoundError(f'PBH data for log10(m_PBH/g)={np.log10(m_PBH):.3f} not found.')
        zero_spec = self.data['phot dNdEdt'][0] * 0.
        self.phot_dNdEdt_interp = interpolate.interp1d(self.data['t'], self.data['phot dNdEdt'], axis=0, bounds_error=False, fill_value=zero_spec) # [phot / eV s BH]
        self.elec_dNdEdt_interp = interpolate.interp1d(self.data['t'], self.data['elec dNdEdt'], axis=0, bounds_error=False, fill_value=zero_spec) # [elec / eV s BH]

        self.M_t = interpolate.interp1d(self.data['t'], self.data['M'], bounds_error=False, fill_value=0) # [g]([s])
        i_start = np.where(self.data['t'] > 1e10)[0][0] # [s]
        dMdt = np.abs(np.gradient(self.data['M'][i_start:], self.data['t'][i_start:])) # [g/s]
        self.dMdt_t = interpolate.interp1d(self.data['t'][i_start:], dMdt, bounds_error=False, fill_value=0) # [g/s]([s])

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
    
    #===== final injection modification =====
    def init_final_inj(self, z_inj_s):
        """Initialize final stage injection parameters if PBH has evaporated by z_end.
        Will be called for DarkHistory and DM21cm.

        Args:
            z_inj_s (array): List of (decreasing) redshifts at which injection happens, plus a final boundary z.

        Set the following attributes:
            z_inj_s (array): As above.
            evaporated_by_end_rs (bool): True if PBH has evaporated by end_rs.
            if evaporated_by_end_rs is True:
                z_final_inj (float): Redshift of the final injection step.
                final_inj_multiplier (float): Injection energy multiplier of the final injection step.
                phot_final_inj_shape (Spectrum): Injection spectral shape of photons at the final injection step.
                elec_final_inj_shape (Spectrum): Injection spectral shape of electrons at the final injection step.
        """
        # Get dM's
        dM_s = []
        for i, z in enumerate(z_inj_s[:-1]):
            t = phys.t_z(z)
            t_next = phys.t_z(z_inj_s[i+1])
            dM = self.dMdt_t(t) * (t_next - t)
            dM_s.append(dM) # dM's length will be one less than z_inj_s

        # Test if BH has evaporated by end_rs
        self.z_inj_s = z_inj_s
        if dM_s[-1] != 0.: # BH has not evaporated
            self.evaporated_by_end = False
            return
        self.evaporated_by_end = True
        self.i_final_inj = np.nonzero(dM_s)[0][-1] # there is a danger of emission spec mismatch with dMdt
        self.z_final_inj = z_inj_s[self.i_final_inj]

        # Final step injection multiplier
        dM_total = self.data['M0'] - self.M_t(phys.t_z(z_inj_s[0])) # [g]
        dM_actual = np.sum(dM_s) # [g]
        dM_extra = np.max(dM_total - dM_actual, 0) # [g]
        self.final_inj_multiplier = (dM_extra + dM_s[self.i_final_inj]) / dM_s[self.i_final_inj]

        # Final step injection spectral shape
        t_final_start = phys.t_z(np.sqrt((1+self.z_final_inj) * (1+self.z_inj_s[self.i_final_inj-1])) - 1) # [s]
        phot_dNdE = self.data['phot dNdEdt'][0] * 0.
        elec_dNdE = self.data['elec dNdEdt'][0] * 0.
        for i, t in enumerate(self.data['t']):
            if t < t_final_start:
                continue
            dt = t - self.data['t'][i-1]
            phot_dNdE += self.data['phot dNdEdt'][i] * dt
            elec_dNdE += self.data['elec dNdEdt'][i] * dt
        self.phot_final_inj_shape = Spectrum(self.abscs['photE'], phot_dNdE, spec_type='dNdE') # normalization does not matter
        self.elec_final_inj_shape = Spectrum(self.abscs['elecEk'], elec_dNdE, spec_type='dNdE')
    
    #===== injections =====
    def n_PBH(self, z):
        """Mean number density of PBHs in [BH / pcm^3]. Constant and includes 'evaporated PBHs'."""
        return phys.rho_DM * (1+z)**3 * self.f_PBH / self.m_eV # [BH / pcm^3]

    def inj_rate(self, z):
        return self.n_PBH(z) * self.inj_per_sec # [inj / pcm^3 s]
    
    def inj_power(self, z):
        return max(1e-100, self.inj_phot_spec(z).toteng() + self.inj_elec_spec(z).toteng()) # [eV / pcm^3 s]

    def inj_phot_spec(self, z, **kwargs):
        """Baseline photon injection spectrum [phot / pcm^3 eV s]."""
        dNdEdt = self.phot_dNdEdt_interp(phys.t_z(z)) * self.n_PBH(z) # [phot / pcm^3 eV s]
        base_spec = Spectrum(self.abscs['photE'], dNdEdt, spec_type='dNdE')
        if self.evaporated_by_end_rs and np.isclose(z, self.z_final_inj):
            inj_eng = base_spec.toteng() * self.final_inj_multiplier
            return self.phot_final_inj_shape / self.phot_final_inj_shape.toteng() * inj_eng
        else:
            return base_spec
        
    def inj_elec_spec(self, z, **kwargs):
        """Baseline electron injection spectrum [elec / pcm^3 eV s]."""
        dNdEdt = self.elec_dNdEdt_interp(phys.t_z(z)) * self.n_PBH(z) # [elec / pcm^3 eV s]
        base_spec = Spectrum(self.abscs['elecEk'], dNdEdt, spec_type='dNdE')
        if self.evaporated_by_end_rs and np.isclose(z, self.z_final_inj):
            inj_eng = base_spec.toteng() * self.final_inj_multiplier
            return self.elec_final_inj_shape / self.elec_final_inj_shape.toteng() * inj_eng
        else:
            return base_spec
    
    def inj_phot_spec_box(self, z, delta_plus_one_box=None, **kwargs):
        return self.inj_phot_spec(z), delta_plus_one_box # [phot / pcm^3 s], [1]

    def inj_elec_spec_box(self, z, delta_plus_one_box=None, **kwargs):
        return self.inj_elec_spec(z), delta_plus_one_box # [elec / pcm^3 s], [1]