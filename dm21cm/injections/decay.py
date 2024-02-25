"""Dark matter decay injection."""

import os
import sys

import numpy as np
import jax.numpy as jnp

sys.path.append(os.environ['DM21CM_DIR'])
import dm21cm.physics as phys
from dm21cm.injections.base import Injection

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec import pppc


class DMDecayInjection (Injection):
    """Dark matter decay injection object.
    
    Args:
        primary (str): Primary injection channel. See darkhistory.pppc.get_pppc_spec
        m_DM (float): DM mass in [eV].
        lifetime (float, optional): Decay lifetime in [s].
    """

    def __init__(self, primary=..., m_DM=..., lifetime=...):
        self.mode = 'DM decay'
        self.primary = primary
        self.m_DM = m_DM
        self.lifetime = lifetime

    def set_binning(self, abscs):
        self.phot_spec_per_inj = pppc.get_pppc_spec(
            self.m_DM, abscs['photE'], self.primary, 'phot', decay=True
        ) # per injection event
        self.elec_spec_per_inj = pppc.get_pppc_spec(
            self.m_DM, abscs['elecEk'], self.primary, 'elec', decay=True
        ) # per injection event

    def is_injecting_elec(self):
        return not np.allclose(self.elec_spec_per_inj.N, 0.)
    
    def get_config(self):
        return {
            'mode': self.mode,
            'primary': self.primary,
            'm_DM': self.m_DM,
            'lifetime': self.lifetime
        }

    #===== injections =====
    def inj_rate_per_Bavg(self, z):
        nBavg = phys.n_B * (1+z)**3 # [Bavg / (physical cm)^3]
        rho_DM = phys.rho_DM * (1+z)**3 # [eV / (physical cm)^3]
        inj_rate = (rho_DM/self.m_DM) / self.lifetime # [inj / (physical cm)^3 s]
        return float(inj_rate / nBavg) # [inj / Bavg s]
    
    def inj_power_per_Bavg(self, z):
        return self.inj_rate_per_Bavg(z) * self.m_DM # [eV / Bavg s]
    
    def inj_phot_spec(self, z, **kwargs):
        return self.phot_spec_per_inj * self.inj_rate_per_Bavg(z) # [ph / Bvg s]
    
    def inj_elec_spec(self, z, **kwargs):
        return self.elec_spec_per_inj * self.inj_rate_per_Bavg(z) # [e / Bvg s]
    
    def inj_phot_spec_box(self, z, delta_plus_one_box=..., **kwargs):
        box_avg = float(jnp.mean(delta_plus_one_box)) # [1] | should be very close to 1
        return self.inj_phot_spec(z) * box_avg, delta_plus_one_box / box_avg # [ph / Bvg s], [1]

    def inj_elec_spec_box(self, z, delta_plus_one_box=..., **kwargs):
        box_avg = float(jnp.mean(delta_plus_one_box))
        return self.inj_elec_spec(z) * box_avg, delta_plus_one_box / box_avg # [e / Bvg s], [1]