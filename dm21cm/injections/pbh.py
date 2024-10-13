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
        self.m_PBH = m_PBH # [g]
        self.f_PBH = f_PBH
        self.inj_per_sec = 1. # [inj / s] | convention: 1 injection event per second

        self.m_eV = (self.m_PBH * u.g * const.c**2).to(u.eV).value # [eV]
        self.n0_PBH = phys.rho_DM * f_PBH / self.m_eV # [BH / pcm^3] | Present day PBH number density

        #----- Load PBH data -----
        try:
            self.data = load_h5_dict(f'{data_dir}/pbh_logm{np.log10(m_PBH):.3f}.h5')
        except FileNotFoundError:
            raise FileNotFoundError(f'PBH data for log10(m_PBH/g)={np.log10(m_PBH):.3f} not found.')
        
        i_start = np.where(self.data['t'] > phys.t_z(5e3))[0][0] # 1e4 is the largest z phys.z_t calculates
        i_end = i_start
        while i_end < len(self.data['t']) and self.data['t'][i_end] < phys.t_z(1): # 1e-8 is the smallest z phys.z_t calculates
            if self.data['t'][i_end] == self.data['t'][i_end-1]:
                break
            i_end += 1
        self.t_s = self.data['t'][i_start:i_end] # [s]
        self.M_s = self.data['M'][i_start:i_end] # [g]
        self.phot_dNdEdt_s = self.data['phot dNdEdt'][i_start:i_end] # [phot / eV s BH]
        self.elec_dNdEdt_s = self.data['elec dNdEdt'][i_start:i_end] # [elec / eV s BH]
        
        self.t_edges = (self.t_s[:-1] + self.t_s[1:]) / 2 # [s]
        self.t_edges = np.concatenate((
            [self.t_s[0] - (self.t_edges[0] - self.t_s[0])],
            self.t_edges,
            [self.t_s[-1] + (self.t_s[-1] - self.t_edges[-1])]
        )) # [s]
        self.z_edges = np.array([phys.z_t(t) for t in self.t_edges])

        #----- Interpolations -----
        self.M_t = interpolate.interp1d(self.t_s, self.M_s, bounds_error=False, fill_value=0) # [g]([s])
        zero_spec = self.data['phot dNdEdt'][0] * 0.
        self.phot_dNdEdt_interp = interpolate.interp1d(self.t_s, self.phot_dNdEdt_s, axis=0, bounds_error=False, fill_value=zero_spec) # [phot / eV s BH]
        self.elec_dNdEdt_interp = interpolate.interp1d(self.t_s, self.elec_dNdEdt_s, axis=0, bounds_error=False, fill_value=zero_spec) # [elec / eV s BH]

        dMdt = np.abs(np.gradient(self.M_s, self.t_s)) # [g/s]
        self.dMdt_t = interpolate.interp1d(self.t_s, dMdt, bounds_error=False, fill_value=0) # [g/s]([s])

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
    # def init_final_inj(self, z_inj_s):
    #     """Initialize final stage injection parameters if PBH has evaporated by z_end.
    #     Will be called for DarkHistory and DM21cm.

    #     Args:
    #         z_inj_s (array): List of (decreasing) redshifts at which injection happens, plus a final boundary z.

    #     Set the following attributes:
    #         z_inj_s (array): As above.
    #         evaporated_by_end_rs (bool): True if PBH has evaporated by end_rs.
    #         if evaporated_by_end_rs is True:
    #             z_final_inj (float): Redshift of the final injection step.
    #             final_inj_multiplier (float): Injection energy multiplier of the final injection step.
    #             phot_final_inj_shape (Spectrum): Injection spectral shape of photons at the final injection step.
    #             elec_final_inj_shape (Spectrum): Injection spectral shape of electrons at the final injection step.
    #     """
    #     # Get dM's
    #     dM_s = []
    #     for i, z in enumerate(z_inj_s[:-1]):
    #         t = phys.t_z(z)
    #         t_next = phys.t_z(z_inj_s[i+1])
    #         dM = self.dMdt_t(t) * (t_next - t)
    #         dM_s.append(dM) # dM's length will be one less than z_inj_s

    #     # Test if BH has evaporated by end_rs
    #     self.z_inj_s = z_inj_s
    #     if dM_s[-1] != 0.: # BH has not evaporated
    #         self.evaporated_by_end = False
    #         return
    #     self.evaporated_by_end = True
    #     self.i_final_inj = np.nonzero(dM_s)[0][-1] # there is a danger of emission spec mismatch with dMdt
    #     self.z_final_inj = z_inj_s[self.i_final_inj]

    #     # Final step injection multiplier
    #     dM_total = self.data['M0'] - self.M_t(phys.t_z(z_inj_s[0])) # [g]
    #     dM_actual = np.sum(dM_s) # [g]
    #     dM_extra = np.max(dM_total - dM_actual, 0) # [g]
    #     self.final_inj_multiplier = (dM_extra + dM_s[self.i_final_inj]) / dM_s[self.i_final_inj]

    #     # Final step injection spectral shape
    #     t_final_start = phys.t_z(np.sqrt((1+self.z_final_inj) * (1+self.z_inj_s[self.i_final_inj-1])) - 1) # [s]
    #     phot_dNdE = self.data['phot dNdEdt'][0] * 0.
    #     elec_dNdE = self.data['elec dNdEdt'][0] * 0.
    #     for i, t in enumerate(self.data['t']):
    #         if t < t_final_start:
    #             continue
    #         dt = t - self.data['t'][i-1]
    #         phot_dNdE += self.data['phot dNdEdt'][i] * dt
    #         elec_dNdE += self.data['elec dNdEdt'][i] * dt
    #     self.phot_final_inj_shape = Spectrum(self.abscs['photE'], phot_dNdE, spec_type='dNdE') # normalization does not matter
    #     self.elec_final_inj_shape = Spectrum(self.abscs['elecEk'], elec_dNdE, spec_type='dNdE')
    

    #===== injections =====
    def n_PBH(self, z_start, z_end=None):
        """Mean physical number density of PBHs in [BH / pcm^3]. Includes 'evaporated PBHs'."""
        return self.n0_PBH * (1+z_start)**3 # [BH / pcm^3]

    def inj_rate(self, z_start, z_end=None):
        return self.n_PBH(z_start, z_end=z_end) * self.inj_per_sec # [inj / pcm^3 s]
    
    def inj_power(self, z_start, z_end=None):
        power = self.inj_phot_spec(z_start, z_end=z_end).toteng() + self.inj_elec_spec(z_start, z_end=z_end).toteng()
        return max(1e-100, power) # [eV / pcm^3 s]

    def inj_phot_spec(self, z_start, z_end=None, **kwargs):
        """Photon injection spectrum [phot / pcm^3 (eV) s]."""
        if z_end is None: # instantaneous rate spectrum
            dndEdt = self.phot_dNdEdt_interp(phys.t_z(z_start)) * self.n_PBH(z_start) # [phot / pcm^3 eV s]
            return Spectrum(self.abscs['photE'], dndEdt, spec_type='dNdE')
        
        t_start = phys.t_z(z_start)
        t_end = phys.t_z(z_end)
        dt = t_end - t_start
        interval_inds = (self.t_edges > t_start) & (self.t_edges < t_end)
        t_edges_interval = np.concatenate(( [t_start], self.t_edges[interval_inds], [t_end] ))
        z_edges_interval = np.concatenate(( [z_start], self.z_edges[interval_inds], [z_end] ))
        dndEdt_s = self.phot_dNdEdt_interp(t_edges_interval) * self.n_PBH(z_edges_interval)[:, None] # [phot / pcm^3 eV s]
        dndE = np.trapz(dndEdt_s, x=t_edges_interval, axis=0) # [phot / pcm^3 eV]
        return Spectrum(self.abscs['photE'], dndE / dt, spec_type='dNdE') # [phot / pcm^3 (eV) s]
        
    def inj_elec_spec(self, z_start, z_end=None, **kwargs):
        """Electron injection spectrum [elec / pcm^3 (eV) s]."""
        if z_end is None: # instantaneous rate spectrum
            dndEdt = self.elec_dNdEdt_interp(phys.t_z(z_start)) * self.n_PBH(z_start) # [elec / pcm^3 eV s]
            return Spectrum(self.abscs['elecEk'], dndEdt, spec_type='dNdE')
        
        t_start = phys.t_z(z_start)
        t_end = phys.t_z(z_end)
        dt = t_end - t_start
        interval_inds = (self.t_edges > t_start) & (self.t_edges < t_end)
        t_edges_interval = np.concatenate(( [t_start], self.t_edges[interval_inds], [t_end] ))
        z_edges_interval = np.concatenate(( [z_start], self.z_edges[interval_inds], [z_end] ))
        dndEdt_s = self.elec_dNdEdt_interp(t_edges_interval) * self.n_PBH(z_edges_interval)[:, None] # [elec / pcm^3 eV s]
        dndE = np.trapz(dndEdt_s, x=t_edges_interval, axis=0) # [elec / pcm^3 eV]
        return Spectrum(self.abscs['elecEk'], dndE / dt, spec_type='dNdE') # [elec / pcm^3 (eV) s]
    
    def inj_phot_spec_box(self, z_start, z_end=None, delta_plus_one_box=None, **kwargs):
        return self.inj_phot_spec(z_start, z_end=z_end), delta_plus_one_box # [phot / pcm^3 (eV) s], [1]

    def inj_elec_spec_box(self, z_start, z_end=None, delta_plus_one_box=None, **kwargs):
        return self.inj_elec_spec(z_start, z_end=z_end), delta_plus_one_box # [elec / pcm^3 (eV) s], [1]