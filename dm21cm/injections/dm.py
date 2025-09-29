"""Dark matter injections."""

import os
import sys

import numpy as np
import jax.numpy as jnp

from darkhistory.spec import pppc

import dm21cm.physics as phys
from dm21cm.injections.base import Injection
from dm21cm.utils import load_h5_dict, abscs
from dm21cm.interpolators import interp1d, bound_action


class DMDecayInjection (Injection):
    """Dark matter decay injection object. See parent class for details.
    
    Args:
        primary (str): Primary injection channel. See darkhistory.pppc.get_pppc_spec
        m_DM (float): DM mass in [eV].
        lifetime (float, optional): Decay lifetime in [s].
    """

    def __init__(self, primary=None, m_DM=None, lifetime=None):
        self.mode = 'DM decay'
        self.primary = primary
        self.m_DM = m_DM
        self.lifetime = lifetime

        self.phot_spec_per_inj = pppc.get_pppc_spec(
            self.m_DM, abscs['photE'], self.primary, 'phot', decay=True
        ) # [phot / inj]
        self.elec_spec_per_inj = pppc.get_pppc_spec(
            self.m_DM, abscs['elecEk'], self.primary, 'elec', decay=True
        ) # [elec / inj]

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
    # Assuming Euler steps. z_end is not used.
    def inj_rate(self, z, z_end=None, **kwargs):
        rho_DM = phys.rho_DM * (1+z)**3 # [eV / pcm^3]
        return float((rho_DM/self.m_DM) / self.lifetime) # [inj / pcm^3 s]
    
    def inj_power(self, z, z_end=None, **kwargs):
        return self.inj_rate(z) * self.m_DM # [eV / pcm^3 s]
    
    def inj_phot_spec(self, z, z_end=None, **kwargs):
        return self.phot_spec_per_inj * self.inj_rate(z) # [phot / pcm^3 s]
    
    def inj_elec_spec(self, z, z_end=None, **kwargs):
        return self.elec_spec_per_inj * self.inj_rate(z) # [elec / pcm^3 s]
    
    def inj_phot_spec_box(self, z, z_end=None, delta_plus_one_box=None, **kwargs):
        return self.inj_phot_spec(z), delta_plus_one_box # [phot / pcm^3 s], [1]

    def inj_elec_spec_box(self, z, z_end=None, delta_plus_one_box=None, **kwargs):
        return self.inj_elec_spec(z), delta_plus_one_box # [elec / pcm^3 s], [1]


class DMPWaveAnnihilationInjection (Injection):
    """DM p-wave annihilation injection object.
    
    Args:
        primary (str): Primary injection channel. See darkhistory.pppc.get_pppc_spec
        m_DM (float): DM mass in [eV].
        c_sigma (float): sigma_v at v=c in [pcm^3/s].
        cell_size (float): Cell size in [cMpc].
        modifier (str, optional): Modifier for the data table.
    """

    def __init__(self, primary=None, m_DM=None, c_sigma=None, cell_size=2., modifier=None):
        self.mode = 'DM p-wave annihilation'
        self.primary = primary
        self.m_DM = m_DM
        self.c_sigma = c_sigma
        self.cell_size = cell_size

        if modifier:
            data_fn = f"/pwave_hmf_summed_rate_{modifier}.h5"
        else:
            data_fn = "/pwave_hmf_summed_rate.h5"
        self.data = load_h5_dict(os.environ['DM21CM_DATA_DIR'] + data_fn) # tables have unit [eV^2 / pcm^3 / cfcm^3]
        self.z_range = self.data['z']
        self.d_range = self.data['d']
        assert self.cell_size == self.data['cell_size'], "Cell size mismatch."

        self.phot_spec_per_inj = pppc.get_pppc_spec(
            self.m_DM, abscs['photE'], self.primary, 'phot', decay=False
        ) # [phot / inj]
        self.elec_spec_per_inj = pppc.get_pppc_spec(
            self.m_DM, abscs['elecEk'], self.primary, 'elec', decay=False
        ) # [elec / inj]

    def is_injecting_elec(self):
        return not np.allclose(self.elec_spec_per_inj.N, 0.)
    
    def get_config(self):
        return {
            'mode': self.mode,
            'primary': self.primary,
            'm_DM': self.m_DM,
            'c_sigma': self.c_sigma,
            'cell_size': self.cell_size
        }
    
    
    #===== injections =====
    def cond_ann_rate_fixed_cell(self, z, delta_plus_one_box):
        """Computes injection rate density with PS halo boost up to fixed cell size."""

        z_in = bound_action(z, self.z_range, 'clip')
        d_box_in = bound_action(delta_plus_one_box - 1, self.d_range, 'clip')

        ps_cond_delta = interp1d(self.data['ps_cond'], self.z_range, z_in)
        ps_cond_box   = interp1d(ps_cond_delta, self.d_range, d_box_in)
        dNtilde_dt_box = ps_cond_box # [eV^2 / pcm^3 ccm^3]
        return dNtilde_dt_box * self.c_sigma / self.m_DM**2 * (1 + z)**3 # [inj / pcm^3 s]

    
    def inj_rate(self, z_start, z_end=None, **kwargs):
        """Instantaneous rate in a homogeneous universe. Use ST table."""
        z_in = bound_action(z_start, self.z_range, 'clip')
        ps_val = interp1d(self.data['ps'], self.z_range, z_in) # [eV^2 / pcm^3 ccm^3]
        return np.clip(np.float32(ps_val * self.c_sigma / self.m_DM**2 * (1 + z_start)**3), 1e-200, None) # [inj / pcm^3 s]
    
    def inj_power(self, z_start, z_end=None, **kwargs):
        """Instantaneous rate in a homogeneous universe. Use ST table."""
        return float(self.inj_rate(z_start) * 2 * self.m_DM) # [eV / pcm^3 s]
    
    def inj_phot_spec(self, z_start, z_end=None, **kwargs):
        """Instantaneous rate in a homogeneous universe. Use ST table."""
        return self.phot_spec_per_inj * float(self.inj_rate(z_start)) # [phot / pcm^3 s]
    
    def inj_elec_spec(self, z_start, z_end=None, **kwargs):
        """Instantaneous rate in a homogeneous universe. Use ST table."""
        return self.elec_spec_per_inj * float(self.inj_rate(z_start)) # [elec / pcm^3 s]
    
    def inj_phot_spec_box(self, z_start, z_end=None, delta_plus_one_box=None, **kwargs):
        self.rate_box = self.cond_ann_rate_fixed_cell(z_start, delta_plus_one_box)
        spec = self.phot_spec_per_inj * float(jnp.mean(self.rate_box))
        weight = self.rate_box / jnp.mean(self.rate_box)
        return spec, weight # [phot / pcm^3 s], [1]

    def inj_elec_spec_box(self, z_start, z_end=None, delta_plus_one_box=None, reuse_rate_box=False, **kwargs):
        if not reuse_rate_box:
            self.rate_box = self.cond_ann_rate_fixed_cell(z_start, delta_plus_one_box)
        spec = self.elec_spec_per_inj * float(jnp.mean(self.rate_box))
        weight = self.rate_box / jnp.mean(self.rate_box)
        return spec, weight # [elec / pcm^3 s], [1]