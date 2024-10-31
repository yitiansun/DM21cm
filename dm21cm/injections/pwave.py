"""P-wave DM injection classes. Including cacher classes which referenced xray data cacher classes."""

import os
import sys
import h5py
import numpy as np

from jax.numpy import fft
import jax.numpy as jnp

sys.path.append(os.environ['DM21CM_DIR'])
import dm21cm.physics as phys
from dm21cm.utils import init_logger
from dm21cm.injections.base import Injection
from dm21cm.utils import load_h5_dict
from dm21cm.interpolators import interp1d, interp1d_vmap, bound_action
from xray_cache import XrayCache

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec import pppc

EPSILON = 1e-6

logger = init_logger(__name__)


#===== Caching =====

class DeltaCache (XrayCache):
    """Cache for density constrast delta.

    Args:
        data_dir (str): Path to the cache directory.
        box_dim (int): Cell number on a side of the box.
        dx (float): The size of each cell [cMpc].

    Attributes:
        isresumed (bool): Whether the cache is resumed from a snapshot. Set to False.
        states (list): List of CachedState objects.
    """

    def __init__(self, data_dir, box_dim=None, dx=None):

        self.data_dir = data_dir
        self.box_cache_path = os.path.join(data_dir, 'delta_cache.h5')

        self.box_dim = box_dim
        self.dx = dx

        self.isresumed = False
        self.init_fft()
    

#===== Injection =====

class DMPWaveAnnihilationInjection (Injection):
    """DM p-wave annihilation injection object.
    
    Args:
        primary (str): Primary injection channel. See darkhistory.pppc.get_pppc_spec
        m_DM (float): DM mass in [eV].
        c_sigma (float): sigma_v at v=c in [pcm^3/s].
        cell_size (float): Cell size in [cMpc].
        ps_r_max (float): Maximum radius of Press-Schetcher calculation [cMpc].
    """

    def __init__(self, primary=None, m_DM=None, c_sigma=None, cell_size=None, ps_r_max=None):
        self.mode = 'DM p-wave annihilation'
        self.primary = primary
        self.m_DM = m_DM
        self.c_sigma = c_sigma
        self.cell_size = cell_size
        self.ps_r_max = ps_r_max

        self.data = load_h5_dict(os.environ['DM21CM_DATA_DIR'] + '/pwave_ann_rate.h5') # tables have unit [eV^2 / cm^6]
        self.z_range = self.data['z_range']
        self.d_range = self.data['delta_range']
        self.r_range = self.data['r_range']
        self.ps_cond_table_rzd = jnp.einsum('zdr->rzd', self.data['ps_cond_ann_rate_table']) # radius, z, delta # TODO: fix data order to rzd

        # initialize fixed cell interpolation data
        # r = self.cell_size / jnp.cbrt(4*jnp.pi/3) # [Mpc] | r of sphere with volume cell_size^3
        # self.ps_cond_table_fixed_cell = interp1d(self.ps_cond_table_rzd, self.data['r_range'], r)

    def set_binning(self, abscs):
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
            'cell_size': self.cell_size,
            'ps_r_max': self.ps_r_max
        }
    
    #===== caching =====
    def init_delta_cache(self, data_dir, box_dim=None, dx=None):
        self.delta_cache = DeltaCache(data_dir, box_dim=box_dim, dx=dx)
    
    #===== injections =====

    def inj_phot_spec_box_past_shell(self, z, state):

        r_start = phys.conformal_dx_between_z(z, state.z_start)
        r_end = phys.conformal_dx_between_z(z, state.z_end)
        r_to_interp = (r_start + r_end) / 2
        z_to_interp = (state.z_start + state.z_end) / 2

        delta_plus_one_box_smoothed = self.delta_cache.get_smoothed_box(state, z)

        # Apply ST-normed PS table
        z_in = bound_action(z_to_interp, self.z_range, 'clip')
        r_in = bound_action(r_to_interp, self.r_range, 'clip')
        delta_in = bound_action(delta_plus_one_box_smoothed - 1, self.d_range, 'clip')

        ps_cond_zd  = interp1d(self.ps_cond_table_rzd, self.r_range, r_in)
        ps_cond_d   = interp1d(ps_cond_zd, self.z_range, z_in)
        ps_cond_box = interp1d_vmap(ps_cond_d, self.d_range, delta_in)
        ps_uncond_val = interp1d(self.data['ps_ann_rate_table'], self.z_range, z_in)
        st_val        = interp1d(self.data['st_ann_rate_table'], self.z_range, z_in)
        dNtilde_dt_box = ps_cond_box * st_val / ps_uncond_val # [eV^2 / ccm^3 pcm^3]
        rate_box = dNtilde_dt_box * self.c_sigma / self.m_DM**2 * (1 + z)**3 # [inj / pcm^3 s]
        rate_box_avg = float(jnp.mean(rate_box)) # [inj / pcm^3 s]

        return state.spectrum * rate_box_avg, rate_box / rate_box_avg # [phot / pcm^3 s], [1]
    

    # def conditional_annihilation_rate(self, z):
    #     """Computes injection rate density with PS halo boost."""

    #     z_range = self.data['z_range']
    #     delta_range = self.data['delta_range']
    #     r_range = self.data['r_range']

    #     d = self.delta_cache.box_dim
    #     total_rate_box = jnp.zeros((d, d, d))

    #     # For each layer, up to largest
    #     for state in self.delta_cache.states[::-1]:
    #         # print(f'> > delta shell {state.z_start:.3f} - {state.z_end:.3f} -> {z:.3f}')
    #         if np.isclose(z, state.z_start):
    #             continue # only need past shells
    #         r_start = phys.conformal_dx_between_z(z, state.z_start)
    #         r_end = phys.conformal_dx_between_z(z, state.z_end)
    #         r_to_interp = (r_start + r_end) / 2
    #         z_to_interp = (state.z_start + state.z_end) / 2
    #         if r_start > self.delta_cache.box_len / 2:
    #             break # don't care about halos larger than the simulation box

    #         # Get smoothed box
    #         delta_plus_one_box_smoothed = self.delta_cache.get_smoothed_box(state, z)
            
    #         # Apply ST-normed PS table: big halo rate
    #         z_in = bound_action(z_to_interp, z_range, 'clip') # emission z
    #         r_in = bound_action(r_to_interp, r_range, 'clip')
    #         delta_in = bound_action(delta_plus_one_box_smoothed - 1, delta_range, 'clip')

    #         ps_cond_zd  = interp1d(self.ps_cond_table_rzd, r_range, r_in)
    #         ps_cond_d   = interp1d(ps_cond_zd, z_range, z_in)
    #         ps_cond_box = interp1d_vmap(ps_cond_d, delta_range, delta_in)
    #         ps_uncond_val = interp1d(self.data['ps_ann_rate_table'], z_range, z_in)
    #         st_val        = interp1d(self.data['st_ann_rate_table'], z_range, z_in)
    #         dNtilde_dt_box = ps_cond_box * st_val / ps_uncond_val # [eV^2 / ccm^3 pcm^3]
    #         total_rate_box += dNtilde_dt_box * self.c_sigma / self.m_DM**2 * (1 + z)**3 # [inj / pcm^3 s]

    #         # Subtract fixed cell contributions
    #         # pass for now

    #     return total_rate_box # [inj / pcm^3 s]

    
    # def cond_ann_rate_fixed_cell(self, z, delta_plus_one_box):
    #     """Computes injection rate density with PS halo boost up to fixed cell size."""

    #     z_range = self.data['z_range']
    #     delta_range = self.data['delta_range']
    #     z_in     = bound_action(z, z_range, 'clip')
    #     delta_in = bound_action(delta_plus_one_box - 1, delta_range, 'clip')

    #     ps_cond_delta = interp1d(self.ps_cond_table_fixed_cell, z_range, z_in)
    #     ps_cond_box   = interp1d_vmap(ps_cond_delta, delta_range, delta_in)
    #     ps_uncond_val = interp1d(self.data['ps_ann_rate_table'], z_range, z_in)
    #     st_val        = interp1d(self.data['st_ann_rate_table'], z_range, z_in)
    #     dNtilde_dt_box = ps_cond_box * st_val / ps_uncond_val # [eV^2 / ccm^3 pcm^3]
    #     return dNtilde_dt_box * self.c_sigma / self.m_DM**2 * (1 + z)**3 # [inj / pcm^3 s]

    
    def inj_rate(self, z_start, z_end=None):
        """Instantaneous rate in a homogeneous universe. Use ST table."""
        z_in = bound_action(z_start, self.z_range, 'clip')
        st_val = interp1d(self.data['st_ann_rate_table'], self.z_range, z_in) # [eV^2 / pcm^6]
        return float(st_val * self.c_sigma / self.m_DM**2) # [inj / pcm^3 s]
    
    def inj_power(self, z_start, z_end=None):
        """Instantaneous rate in a homogeneous universe. Use ST table."""
        return float(self.inj_rate(z_start) * 2 * self.m_DM) # [eV / pcm^3 s]
    
    def inj_phot_spec(self, z_start, z_end=None, **kwargs):
        """Instantaneous rate in a homogeneous universe. Use ST table."""
        return self.phot_spec_per_inj * float(self.inj_rate(z_start)) # [phot / pcm^3 s]
    
    def inj_elec_spec(self, z_start, z_end=None, **kwargs):
        """Instantaneous rate in a homogeneous universe. Use ST table."""
        return self.elec_spec_per_inj * float(self.inj_rate(z_start)) # [elec / pcm^3 s]
    
    def inj_phot_spec_box(self, z_start, z_end=None, delta_plus_one_box=None, **kwargs):
        self.rate_box = self.cond_ann_rate_fixed_cell(z, delta_plus_one_box) + self.cond_ann_rate_big_halo(z, delta_plus_one_box)
        spec = self.phot_spec_per_inj * float(jnp.mean(self.rate_box))
        weight = self.rate_box / jnp.mean(self.rate_box)
        return spec, weight # [phot / pcm^3 s], [1]

    def inj_elec_spec_box(self, z_start, z_end=None, delta_plus_one_box=None, reuse_rate_box=False, **kwargs):
        if not reuse_rate_box:
            self.rate_box = self.cond_ann_rate_fixed_cell(z, delta_plus_one_box) + self.cond_ann_rate_big_halo(z, delta_plus_one_box)
        spec = self.elec_spec_per_inj * float(jnp.mean(self.rate_box))
        weight = self.rate_box / jnp.mean(self.rate_box)
        return spec, weight # [elec / pcm^3 s], [1]