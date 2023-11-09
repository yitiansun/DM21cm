"""Xray data cacher classes."""

import os
import sys
import h5py
import numpy as np

from jax import config
config.update("jax_enable_x64", True)

sys.path.append("..")
import dm21cm.physics as phys
from dm21cm.spectrum import AttenuatedSpectrum

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.spectrum import Spectrum


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

    def __init__(self, data_path, cosmo, N, dx, shell_Rmax, Rmax):

        self.data_path = data_path
        self.cosmo = cosmo
        self.N = N
        self.dx = dx
        self.shell_Rmax = shell_Rmax
        self.Rmax = Rmax

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
        self.spectrum_cache.advance_spectrum(attenuation_factor, z)

    def get_smoothing_radii(self, z_receiver, z1, z2):
        """Evaluates the shell radii [cfMpc] for a receiver at `z_receiver` for emission between redshifts `z1` and `z2`."""
        R1 = np.abs(phys.conformal_dt_between_z(z_receiver, z1)) * phys.c / phys.Mpc
        R2 = np.abs(phys.conformal_dt_between_z(z_receiver, z2)) * phys.c / phys.Mpc
        return np.sort([R1, R2])

    def smooth_box(self, box, R1, R2):
        """Smooths the box with a top-hat window function.

        Args:
            box (array): The box to smooth.
            R1 (float): The inner radius of the smoothing window [cfMpc].
            R2 (float): The outer radius of the smoothing window [cfMpc].

        Returns:
            (array, bool): The smoothed box, whether the whole box is averaged.
        """
        R1 = 1e-10
        if R2 > self.shell_Rmax:
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

        # is_box_averaged = max(R1, R2) > self.Rmax #[cfMpc]
        return smoothed_box, spectrum, is_box_averaged, z_donor, min(self.Rmax-1e-6, max(R1, R2))


class SpectrumCache:
    """Cache for the X-ray spectra."""
    
    def __init__(self):
        self.spectrum_list = []
        self.z_s = np.array([])
        self.low_E_cutoff = 500. # [eV]
        
    def cache_spectrum(self, spec, z):
        self.spectrum_list.append(spec)
        self.z_s = np.append(self.z_s, z)

    def clear_cache(self):
        self.spectrum_list = []
        self.z_s = np.array([])
        
    def attenuate(self, attenuation_arr):
        for spec in self.spectrum_list:
            if isinstance(spec, Spectrum):
                spec.switch_spec_type('N')
                spec.N *= attenuation_arr
            elif isinstance(spec, AttenuatedSpectrum):
                spec.switch_spec_type('N')
                spec.attenuate(attenuation_arr)
            else:
                raise TypeError('Spectrum type not recognized.')
            
    def redshift(self, z_target):
        for spec in self.spectrum_list:
            spec.switch_spec_type('N')
            spec.redshift(1+z_target)

    def cutoff(self):
        for spec in self.spectrum_list:
            spec.switch_spec_type('N')
            i_low = np.searchsorted(spec.eng, self.low_E_cutoff)
            if isinstance(spec, Spectrum):
                spec.N[:i_low] *= 0.
            elif isinstance(spec, AttenuatedSpectrum):
                spec.spectrum.N[:i_low] *= 0.
            else:
                raise TypeError('Spectrum type not recognized.')

    def advance_spectrum(self, attenuation_factor, z):
        self.attenuate(attenuation_factor)
        self.redshift(z)
        self.cutoff()
            
    def get_spectrum(self, z_target):
        spec_index = np.argmin(np.abs(self.z_s - z_target))
        return self.spectrum_list[spec_index]

    def toteng_arr(self):
        return np.array([spec.toteng() for spec in self.spectrum_list])

    def total_spec(self):
        assert len(self.spectrum_list) > 0
        N = np.zeros_like(self.spectrum_list[0].N)
        for spec in self.spectrum_list:
            N += spec.N
        return Spectrum(self.spectrum_list[0].eng, N, spec_type='N', rs=1+np.min(self.z_s))


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
            box = jnp.array(archive['Box_' + str(box_index)], dtype=jnp.complex64)
        
        return box
