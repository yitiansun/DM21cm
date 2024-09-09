"""P-wave DM injection classes. Including cacher classes which referenced xray data cacher classes."""

import os
import sys
import h5py
import numpy as np

sys.path.append(os.environ['DM21CM_DIR'])
import dm21cm.physics as phys
from dm21cm.utils import init_logger
from dm21cm.injections.base import Injection
from dm21cm.utils import load_h5_dict
from dm21cm.interpolators import interp1d, interp1d_vmap, bound_action

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec import pppc


USE_JAX_FFT = True
if USE_JAX_FFT:
    from jax.numpy import fft
    import jax.numpy as jnp
else:
    from numpy import fft
    jnp = np

EPSILON = 1e-6

logger = init_logger(__name__)


#===== Caching =====

class CachedDeltaBox:
    """Cached delta box object.

    Args:
        key (str):           Key to box in cache file.
        z_start (float):     The starting redshift of the step that hosts the emissivity.
        z_end (float):       The ending redshift of the step, at which the spectrum is saved.
    """

    def __init__(self, key, z_start, z_end):
        self.key = key # key to box in the cache file
        self.z_start = z_start
        self.z_end = z_end

    def append_box(self, hf, box, overwrite=False):
        """Fourier transform and append the box to the cache file."""
        if overwrite and self.key in hf:
            del hf[self.key]
        hf.create_dataset(self.key, data=fft.rfftn(box))

    def get_ftbox(self, hf):
        """Get the Fourier transformed box from the cache file."""
        return jnp.array(hf[self.key][()], dtype=jnp.complex64)
    

class DeltaCache:
    """Cache for delta.

    Args:
        data_dir (str): Path to the cache directory.
        box_dim (int): The dimension of the box.
        dx (float): The size of each cell [cfMpc].

    Attributes:
        box_len (float): Length of box [cfMpc].
        states (list): List of CachedDeltaBox objects.
        z_start_end_list (list): List of z_start and z_end for CachedDeltaBox.
    """

    def __init__(self, data_dir, box_dim=None, dx=None):

        self.data_dir = data_dir
        self.box_cache_path = os.path.join(data_dir, 'delta_box_cache.h5')

        self.box_dim = box_dim
        self.dx = dx
        self.box_len = self.box_dim * self.dx

        self.states = []
        self.z_start_end_list = []
        self.init_fft()

    def init_fft(self):
        k = fft.fftfreq(self.box_dim, d=self.dx)
        kReal = fft.rfftfreq(self.box_dim, d=self.dx)
        self.kMag = 2*jnp.pi*jnp.sqrt(k[:, None, None]**2 + k[None, :, None]**2 + kReal[None, None, :]**2)
        self.kMag = jnp.clip(self.kMag, EPSILON, None)

    def cache(self, z_start, z_end, box):
        state = CachedDeltaBox(f'z{z_start:.3f}_z{z_end:.3f}', z_start, z_end)
        with h5py.File(self.box_cache_path, 'a') as hf:
            state.append_box(hf, box, overwrite=False)
        self.states.append(state)
        self.z_start_end_list.append((z_start, z_end))

    def clear_cache(self):
        self.states = []
        if os.path.exists(self.box_cache_path):
            os.remove(self.box_cache_path)

    def smooth_box(self, ftbox, R1, R2):
        """Smooths the box with a top-hat shell window function.

        Args:
            ftbox (array): The Fourier transformed box to smooth.
            R1 (float): The inner radius of the smoothing window [cfMpc].
            R2 (float): The outer radius of the smoothing window [cfMpc].

        Returns:
            array: The smoothed box.
        """
        # Volumetric weighting factors for combining the window functions
        R1, R2 = np.sort([R1, R2])
        w1 = R1**3 / (R2**3 - R1**3)
        w2 = R2**3 / (R2**3 - R1**3)
        R1 = np.clip(R1, EPSILON, None)
        R2 = np.clip(R2, EPSILON, None)

        # Construct the smoothing functions in the frequency domain
        W1 = 3*(jnp.sin(self.kMag*R1) - self.kMag*R1 * jnp.cos(self.kMag*R1)) / (self.kMag*R1)**3
        W2 = 3*(jnp.sin(self.kMag*R2) - self.kMag*R2 * jnp.cos(self.kMag*R2)) / (self.kMag*R2)**3
        if USE_JAX_FFT:
            W1.at[0, 0, 0].set(1.)
            W2.at[0, 0, 0].set(1.)
        else:
            W1[0, 0, 0] = 1.
            W2[0, 0, 0] = 1.

        # Combine the window functions
        W = w2*W2 - w1*W1

        return fft.irfftn(ftbox * W)
    
    def get_smoothed_box(self, state, z_receiver):
        """Get the smoothed box w.r.t. the receiver redshift.
        
        Args:
            state (CachedState): The donor shell's state.
            z_receiver (float): The redshift of the receiver.

        Returns:
            array: The smoothed box.
        """

        r_start = phys.conformal_dx_between_z(state.z_start, z_receiver)
        r_end   = phys.conformal_dx_between_z(state.z_end,   z_receiver)
        with h5py.File(self.box_cache_path, 'r') as hf:
            smoothed_box = self.smooth_box(state.get_ftbox(hf), r_start, r_end)
        return smoothed_box
    

#===== Injection =====

class DMPWaveAnnihilationInjection (Injection):
    """DM p-wave annihilation injection object.
    
    Args:
        primary (str): Primary injection channel. See darkhistory.pppc.get_pppc_spec
        m_DM (float): DM mass in [eV].
        c_sigma (float): sigma_v at v=c in [pcm^3/s].
        cell_size (float): Cell size in [cMpc].
        big_halo_contribution (bool): Whether to include contribution of halos whose progenitors are larger than a pixel.
    """

    def __init__(self, primary=None, m_DM=None, c_sigma=None, cell_size=None, big_halo_contribution=False):
        self.mode = 'DM p-wave annihilation'
        self.primary = primary
        self.m_DM = m_DM
        self.c_sigma = c_sigma
        self.cell_size = cell_size
        self.big_halo_contribution = big_halo_contribution

        self.data = load_h5_dict(os.environ['DM21CM_DATA_DIR'] + '/pwave_ann_rate.h5')
        # tables have unit [eV^2 / cm^6]
        # initialize fixed cell interpolation data
        r = self.cell_size / jnp.cbrt(4*jnp.pi/3) # [Mpc] | r of sphere with volume cell_size^3
        data_rzd = jnp.einsum('zdr->rzd', self.data['ps_cond_ann_rate_table']) # radius, z, delta
        self.ps_cond_table_fixed_cell = interp1d(data_rzd, self.data['r_range'], r)

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
            'big_halo_contribution': self.big_halo_contribution,
        }
    
    #===== caching =====
    def init_delta_cache(self, data_dir, box_dim=None, dx=None):
        if not self.big_halo_contribution:
            return
        self.delta_cache = DeltaCache(data_dir, box_dim=box_dim, dx=dx)
    
    #===== injections =====
    def cond_ann_rate_big_halo(self, z, delta_plus_one_box):
        """Computes injection rate density with PS halo boost for halos larger than fixed cell size.
        Fixed cell contributions are subtracted."""

        # For each layer, up to largest
        for z_start, z_end in self.delta_cache.z_start_list[::-1]:
            if np.isclose(z, z_start):
                continue # only need past shells
            r_start = phys.conformal_dx_between_z(z, z_start)
            r_end = phys.conformal_dx_between_z(z, z_end)
            r_to_interp = (r_start + r_end) / 2
            if r_start > self.delta_cache.box_len / 2:
                break # don't care about halos larger than the simulation box
        # Get smoothed box
        # Apply ST-normed PS table: big halo rate
        # Subtract cutoff at fixed cell scale.
        # return
        return box # [inj / pcm^3 s]
    
    def cond_ann_rate_fixed_cell(self, z, delta_plus_one_box):
        """Computes injection rate density with PS halo boost up to fixed cell size."""
        z_in = bound_action(z, self.data['z_range'], 'clip')
        delta_in = bound_action(delta_plus_one_box - 1, self.data['delta_range'], 'clip')
        ps_cond_delta = interp1d(self.ps_cond_table_fixed_cell, self.data['z_range'], z_in)
        ps_cond_box = interp1d_vmap(ps_cond_delta, self.data['delta_range'], delta_in)
        ps_uncond_val = interp1d(self.data['ps_ann_rate_table'], self.data['z_range'], z_in)
        st_val = interp1d(self.data['st_ann_rate_table'], self.data['z_range'], z_in)
        dNtilde_dt_box = ps_cond_box * st_val / ps_uncond_val # [eV^2 / ccm^3 pcm^3]
        return dNtilde_dt_box * self.c_sigma / self.m_DM**2 * (1 + z)**3 # [inj / pcm^3 s]
    
    def inj_rate(self, z):
        z_in = bound_action(z, self.data['z_range'], 'clip')
        # for rate in a uniform universe, we just use ST table
        st_val = interp1d(self.data['st_ann_rate_table'], self.data['z_range'], z_in) # [eV^2 / pcm^6]
        return float(st_val * self.c_sigma / self.m_DM**2) # [inj / pcm^3 s]
    
    def inj_power(self, z):
        return float(self.inj_rate(z) * 2 * self.m_DM) # [eV / pcm^3 s]
    
    def inj_phot_spec(self, z, **kwargs):
        return self.phot_spec_per_inj * float(self.inj_rate(z)) # [phot / pcm^3 s]
    
    def inj_elec_spec(self, z, **kwargs):
        return self.elec_spec_per_inj * float(self.inj_rate(z)) # [elec / pcm^3 s]
    
    def inj_phot_spec_box(self, z, delta_plus_one_box=None, **kwargs):
        self.rate_box = self.cond_ann_rate_fixed_cell(z, delta_plus_one_box) + self.cond_ann_rate_big_halo(z, delta_plus_one_box)
        spec = self.phot_spec_per_inj * float(jnp.mean(self.rate_box))
        weight = self.rate_box / jnp.mean(self.rate_box)
        return spec, weight # [phot / pcm^3 s], [1]

    def inj_elec_spec_box(self, z, delta_plus_one_box=None, reuse_rate_box=False, **kwargs):
        if not reuse_rate_box:
            self.rate_box = self.cond_ann_rate_fixed_cell(z, delta_plus_one_box) + self.cond_ann_rate_big_halo(z, delta_plus_one_box)
        spec = self.elec_spec_per_inj * float(jnp.mean(self.rate_box))
        weight = self.rate_box / jnp.mean(self.rate_box)
        return spec, weight # [elec / pcm^3 s], [1]