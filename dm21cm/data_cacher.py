"""Xray data cacher classes."""

import os
import sys
import h5py
import numpy as np

sys.path.append("..")
import dm21cm.physics as phys


USE_JAX_FFT = True
if USE_JAX_FFT:
    from jax.numpy import fft
    import jax.numpy as jnp
else:
    from numpy import fft
    jnp = np


class Cacher:
    """Data cacher for xray.

    Args:
        data_path (str): Path to the HDF5 cache file.
        cosmo (astropy.cosmology): Cosmology.
        N (int): Number of grid points.
        dx (float): Grid spacing.

    Notes:
        Brightness = energy per averaged baryon.
    """

    def __init__(self, data_path, cosmo, N, dx, xraycheck=False):

        self.data_path = data_path
        self.cosmo = cosmo
        self.N = N
        self.dx = dx
        self.xraycheck = xraycheck

        # Generate the k magnitudes and save them
        k = fft.fftfreq(N, d = dx)
        kReal = fft.rfftfreq(N, d = dx)
        self.kMag = 2*jnp.pi*jnp.sqrt(k[:, None, None]**2 + k[None, :, None]**2 + kReal[None, None, :]**2)
    
        self.spectrum_cache = SpectrumCache()
        self.brightness_cache = BrightnessCache(self.data_path)

    def cache(self, z, box, spec):
        self.spectrum_cache.cache_spectrum(spec, z)
        self.brightness_cache.cache_box(box, z)

    def clear_cache(self):
        self.spectrum_cache.clear_cache()
        self.brightness_cache.clear_cache()

    def advance_spectrum(self, attenuation_factor, z):
        self.spectrum_cache.attenuate(attenuation_factor)
        self.spectrum_cache.redshift(z) 

    def get_smoothing_radii(self, z_receiver, z1, z2):
        """Evaluates the shell radii [cfMpc] for a receiver at `z_receiver` for emission between redshifts `z1` and `z2`."""
        R1 = np.abs(phys.conformal_dt_between_z(z_receiver, z1)) * phys.c / phys.Mpc
        R2 = np.abs(phys.conformal_dt_between_z(z_receiver, z2)) * phys.c / phys.Mpc
        return R1, R2

    def smooth_box(self, box, R1, R2):
        """Smooths the box with a top-hat window function.

        Args:
            box (array): The box to smooth.
            R1 (float): The inner radius of the smoothing window [cfMpc].
            R2 (float): The outer radius of the smoothing window [cfMpc].

        Returns:
            (array, bool): The smoothed box, whether the whole box is averaged.
        """
        if self.xraycheck:
            R1 = 1e-10
        if min(R1, R2) > self.N // 2 * self.dx:
            box = fft.irfftn(box)
            is_box_averaged = True
            return jnp.mean(box) * jnp.ones_like(box), is_box_averaged

        # Volumetric weighting factors for combining the window functions
        R1, R2 = np.sort([R1, R2])
        w1 = R1**3 / (R2**3 - R1**3)
        w2 = R2**3 / (R2**3 - R1**3)
        R1 = np.clip(R1, 1e-6, None)
        R2 = np.clip(R2, 1e-6, None)
        self.kMag = jnp.clip(self.kMag, 1e-6, None)

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
        del W1, W2

        # Multiply by flux_factor. `box` is now the equivalent universe density of photons.
        box = fft.irfftn(box * W)
        is_box_averaged = False
        return box, is_box_averaged

    def get_annulus_data(self, z_receiver, z_donor, z_next_donor):

	    # Get the donor box
        box_index = np.argmin(np.abs(self.brightness_cache.z_s - z_donor))
        box = self.brightness_cache.get_box(box_index)

        # Get the smoothing radii in comoving coordinates and canonically sort them
        R1, R2 = self.get_smoothing_radii(z_receiver, z_donor, z_next_donor)

        # Perform the box smoothing operation 
        smoothed_box, is_box_averaged = self.smooth_box(box, R1, R2)

        # Get the spectrum
        spectrum = self.spectrum_cache.get_spectrum(z_donor)
        if self.xraycheck:
            is_box_averaged = max(R1, R2) > 500. #[cfMpc]
            return smoothed_box, spectrum, is_box_averaged, z_donor, min(512.-1e-6, max(R1, R2))
        else:
            return smoothed_box, spectrum, is_box_averaged


class SpectrumCache:
    """Cache for the X-ray spectra."""
    
    def __init__(self):
        self.spectrum_list = []
        self.z_s = np.array([])
        
    def cache_spectrum(self, spec, z):
        self.spectrum_list.append(spec)
        self.z_s = np.append(self.z_s, z)

    def clear_cache(self):
        self.spectrum_list = []
        self.z_s = np.array([])
        
    def attenuate(self, attenuation_arr):
        for spec in self.spectrum_list:
            spec.switch_spec_type('N')
            spec.N *= attenuation_arr
            
    def redshift(self, z_target):
        for spec in self.spectrum_list:
            spec.switch_spec_type('N')
            spec.redshift(1+z_target)
            
    def get_spectrum(self, z_target):
        spec_index = np.argmin(np.abs(self.z_s - z_target))
        return self.spectrum_list[spec_index]


class BrightnessCache:
    """Cache for the X-ray brightness boxes.

    Args:
        data_path (str): Path to the HDF5 cache file.

    Notes:
        Brightness = energy per averaged baryon.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.z_s = np.array([])
        self.box_mean_s = np.array([])

    def clear_cache(self):
        if os.path.exists(self.data_path):
            os.remove(self.data_path)
        self.z_s = np.array([])
        self.box_mean_s = np.array([])

    def cache_box(self, box, z):
        """Adds the X-ray box box at the specified redshift to the cache.

        Args:
            box (np.ndarray): The X-ray brightness box to cache. (photons / Mpccm^3)
            z (float): The redshift of the box.
        """

        box_index = len(self.z_s)

        with h5py.File(self.data_path, 'a') as archive:
            archive.create_dataset('Box_' + str(box_index), data = fft.rfftn(box))
            
        self.z_s = np.append(self.z_s, z)
        self.box_mean_s = np.append(self.box_mean_s, jnp.mean(box))

    def get_box(self, z):
        """Returns the brightness box and spectrum at the specified cached state."""

	    # Get the index of the donor box
        box_index = np.argmin(np.abs(self.z_s - z))

        with h5py.File(self.data_path, 'r') as archive:
            box = np.array(archive['Box_' + str(box_index)], dtype = complex)
        
        return box
