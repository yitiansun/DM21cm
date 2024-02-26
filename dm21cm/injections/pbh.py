"""Primordial Black Hole (PBH) injection."""

import os
import sys

import numpy as np
from scipy import interpolate
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.spectrum import Spectrum

sys.path.append(os.environ['DM21CM_DIR'])
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

        self.raw_phot_dN_dEdt_per_BH = ... # [N / BH eV s]
        self.raw_elec_dN_dEdt_per_BH = ... # [N / BH eV s]
        self.t_arr = ... # [s]
        self.eng_arr = ... # [eV]

    def set_binning(self, abscs):
        self.phot_dNdt_table = []
        for raw_spec in self.raw_phot_dN_dEdt_per_BH:
            spec = Spectrum(self.eng_arr, raw_spec, spec_type='dNdE')
            spec.switch_spec_type('N') # [N / BH s]
            spec.rebin_fast(abscs['photE'])
            self.phot_dNdt_table.append(spec.N)
        self.phot_dNdt_interp = interpolate.interp1d(self.t_arr, self.phot_dNdt_table, axis=0) # [N / BH s]

        self.elec_dNdt_table = []
        for raw_spec in self.raw_elec_dN_dEdt_per_BH:
            ind_first = np.where(self.eng_arr > phys.m_e)[0][0]
            spec = Spectrum(self.eng_arr[ind_first:] - phys.m_e, raw_spec[ind_first:], spec_type='dNdE')
            spec.switch_spec_type('N') # [N / BH s]
            spec.rebin_fast(abscs['elecEk'])
            self.elec_dNdt_table.append(spec.N)
        self.elec_dNdt_interp = interpolate.interp1d(self.t_arr, self.elec_dNdt_table, axis=0) # [N / BH s]

    def is_injecting_elec(self):
        return not np.allclose(self.elec_dNdt_table, 0.)
    
    def get_config(self):
        return {
            'mode': self.mode,
            'm_PBH': self.m_PBH,
            'f_PBH': self.f_PBH
        }
    
    #===== injections =====
    def N_PBH_per_Bavg(self, z):
        return phys.rho_DM * (1+z)**3 * self.f_PBH / self.m_PBH # [BH / Bavg]

    def inj_rate_per_Bavg(self, z):
        # convention: 1 injection event per second
        return self.N_PBH_per_Bavg(z) # [inj / Bavg s]
    
    def inj_power_per_Bavg(self, z):
        pass

    def inj_phot_spec(self, z, **kwargs):
        t = cosmo.age(z).to(u.s).value # [s]
        N_arr = self.phot_dNdt_interp(t) * self.inj_rate_per_Bavg