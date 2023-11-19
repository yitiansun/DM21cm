"""Xray data cacher classes."""

import os
import sys
import h5py
import pickle
import logging
import numpy as np

sys.path.append(os.environ['DM21CM_DIR'])
import dm21cm.physics as phys

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.spectrum import Spectrum


USE_JAX_FFT = True
if USE_JAX_FFT:
    from jax.numpy import fft
    import jax.numpy as jnp
else:
    from numpy import fft
    jnp = np

EPSILON = 1e-6


class CachedState:
    """Cached data for xray spectrum and emissivity.

    Args:
        z_start (float):     The starting redshift of the step that hosts the emissivity.
        z_end (float):       The ending redshift of the step, at which the spectrum is saved.
        spectrum (Spectrum): The X-ray spectrum in units of photon per averaged baryon.
        box (array):         The X-ray relative emissivity box.
        isinbath (bool):     Whether the spectrum has been deposited to the bath.
    """

    def __init__(self, key, z_start, z_end, spectrum):

        self.key = key # key to box in the cache file
        self.z_start = z_start
        self.z_end = z_end
        self.spectrum = spectrum
        self.spectrum.switch_spec_type('N')
        self.isinbath = False

    def append_box(self, hf, box):
        hf.create_dataset(self.key, data=fft.rfftn(box)) # hf should be in append mode

    def get_ftbox(self, hf):
        return jnp.array(hf[self.key][()], dtype=jnp.complex64)

    def attenuate(self, attenuation_arr):
        self.spectrum.N *= attenuation_arr

    def redshift(self, z_target):
        self.spectrum.redshift(1+z_target)


class XrayCache:
    """Cache for xray.

    Args:
        data_dir (str): Path to the cache directory.
        box_dim (int): The dimension of the box.
        dx (float): The size of each cell [cfMpc].
    """

    def __init__(self, data_dir, box_dim=None, dx=None, load_snapshot=False):

        self.data_dir = data_dir
        self.box_cache_path = os.path.join(data_dir, 'xray_box_cache.h5')
        self.snapshot_path = os.path.join(data_dir, 'xray_cache_snapshot.p')

        self.box_dim = box_dim
        self.dx = dx
        if load_snapshot:
            if os.path.exists(self.snapshot_path):
                self.load_snapshot()
                z_latest = self.states[-1].z_end if len(self.states) > 0 else jnp.nan
                logging.warning(f'Resuming from snapshot at {self.snapshot_path} with latest redshift z={z_latest:.3f}.')
            else:
                logging.warning(f'No snapshot found at {self.snapshot_path}, restarting run.')
                self.states = []
        else:
            self.states = []

        self.init_fft()

    def init_fft(self):
        k = fft.fftfreq(self.box_dim, d=self.dx)
        kReal = fft.rfftfreq(self.box_dim, d=self.dx)
        self.kMag = 2*jnp.pi*jnp.sqrt(k[:, None, None]**2 + k[None, :, None]**2 + kReal[None, None, :]**2)
        self.kMag = jnp.clip(self.kMag, EPSILON, None)

    def cache(self, z_start, z_end, spectrum, box):
        state = CachedState(f'z{z_start:.3f}_z{z_end:.3f}', z_start, z_end, spectrum)
        with h5py.File(self.box_cache_path, 'a') as hf:
            state.append_box(hf, box)
        self.states.append(state)

    def clear_cache(self):
        self.states = []
        if os.path.exists(self.box_cache_path):
            os.remove(self.box_cache_path)
        if os.path.exists(self.snapshot_path):
            os.remove(self.snapshot_path)

    def save_snapshot(self):
        pickle.dump(self.states, open(self.snapshot_path, 'wb'))

    def load_snapshot(self):
        self.states = pickle.load(open(self.snapshot_path, 'rb'))

    @property
    def i_shell_start(self):
        isinbath_arr = np.array([state.isinbath for state in self.states])
        return np.where(isinbath_arr == False)[0][0] if len(isinbath_arr) > 0 else 0

    # @property
    # def z_s(self):
    #     return np.array([state.z_end for state in self.states])

    # def ind(self, z):
    #     """Get index closest to z with bounds checking."""
    #     z_s = self.z_s
    #     atol = 1e-3
    #     if not z > np.min(z_s) - atol and z < np.max(z_s) + atol:
    #         raise ValueError(f'z={z} out of bounds {np.min(z_s)} - {np.max(z_s)}.')
    #     return np.argmin(np.abs(z_s - z))

    # def get_state(self, z):
    #     """Get the cached state with redshift closest to z."""
    #     return self.states[self.ind(z)]

    def advance_spectra(self, attenuation_arr, z_target):
        """Attenuate and redshift the spectra of states to the target redshift."""
        for state in self.states:
            state.attenuate(attenuation_arr)
            state.redshift(z_target)

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

        r_start = phys.conformal_dx_between_z(state.z_start, z_receiver)
        r_end   = phys.conformal_dx_between_z(state.z_end,   z_receiver)
        with h5py.File(self.box_cache_path, 'r') as hf:
            smoothed_box = self.smooth_box(state.get_ftbox(hf), r_start, r_end)
        return smoothed_box
    
    def is_latest_z(self, z):
        if len(self.states) == 0:
            return True
        return np.isclose(z, self.states[-1].z_end)