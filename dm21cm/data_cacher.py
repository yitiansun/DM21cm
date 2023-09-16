"""Xray data cacher classes."""

import sys
import h5py
import numpy as np

sys.path.append("..")
import dm21cm.physics as phys


USE_JAX_FFT = True
if USE_JAX_FFT:
    from jax.numpy import fft
else:
    from numpy import fft


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

    def __init__(self, data_path, cosmo, N, dx):

        self.cosmo = cosmo
        self.N = N
        self.dx = dx

        # Generate the kmagnitudes and save them
        k = fft.fftfreq(N, d = dx)
        kReal = fft.rfftfreq(N, d = dx)
        self.kMag = 2*np.pi*np.sqrt(k[:, None, None]**2 + k[None, :, None]**2 + kReal[None, None, :]**2)
    
        self.spectrum_cache = SpectrumCache()
        self.brightness_cache = BrightnessCache(data_path)

    def set_cache(self, z, box, spec):
        self.spectrum_cache.cache_spectrum(spec, z)
        self.brightness_cache.cache_box(box, z)

    def advance_spectrum(self, attenuation_factor, z):
        self.spectrum_cache.attenuate(attenuation_factor)
        self.spectrum_cache.redshift(z) 

    def get_smoothing_radii(self, z_receiver, z1, z2):
        """
        Evaluates the shell radii for a receiver at `z_receiver` for emission between redshifts
        `z1` and `z2`
        """
        R1_Mpc = np.abs(phys.conformal_dt_between_z(z_receiver, z1)) * phys.c / phys.Mpc
        R2_Mpc = np.abs(phys.conformal_dt_between_z(z_receiver, z2)) * phys.c / phys.Mpc

        return R1_Mpc, R2_Mpc

    def smooth_box(self, box, R1, R2):

        if min(R1, R2) > self.N // 2 * self.dx:
            box = fft.irfftn(box)
            return np.mean(box) * np.ones_like(box), True

        # Volumetric weighting factors for combining the window functions
        R1, R2 = np.sort([R1, R2])
        w1 = R1**3 / (R2**3 - R1**3)
        w2 = R2**3 / (R2**3 - R1**3)
        R1 = np.clip(R1, 1e-6, None)
        R2 = np.clip(R2, 1e-6, None)
        self.kMag = np.clip(self.kMag, 1e-6, None)

        # Construct the smoothing functions in the frequency domain
        W1 = 3*(np.sin(self.kMag*R1) - self.kMag*R1 * np.cos(self.kMag*R1)) / (self.kMag*R1)**3
        W2 = 3*(np.sin(self.kMag*R2) - self.kMag*R2 * np.cos(self.kMag*R2)) / (self.kMag*R2)**3
        W1[0, 0, 0] = 1
        W2[0, 0, 0] = 1

        # Combine the window functions
        W = w2*W2 - w1*W1
        del W1, W2

        # Multiply by flux_factor. `box` is now the equivalent universe density of photons.
        box = fft.irfftn(box * W)

        return box, False

    def get_annulus_data(self, z_receiver, z_donor, z_next_donor):

	    # Get the donor box
        box_index = np.argmin(np.abs(self.brightness_cache.redshifts - z_donor))
        box = self.brightness_cache.get_box(box_index)

        # Get the smoothing radii in comoving coordinates and canonically sort them
        R1, R2 = self.get_smoothing_radii(z_receiver, z_donor, z_next_donor)

        # Perform the box smoothing operation 
        smoothed_box, type_bool = self.smooth_box(box, R1, R2)

        # Get the spectrum
        spectrum = self.spectrum_cache.get_spectrum(z_donor)
        return smoothed_box, spectrum, type_bool


class SpectrumCache:
    
    def __init__(self):
        self.spectrum_list = []
        self.z_s = np.array([])
        
    def cache_spectrum(self, spec, z):
        self.spectrum_list.append(spec)
        self.z_s = np.append(self.z_s, z)
        
    def attenuate(self, attenuation_factor):
        for spec in self.spectrum_list:
            spec.N *= attenuation_factor
            
    def redshift(self, z_target):
        for spec in self.spectrum_list:
            spec.redshift(1+z_target)
            
    def get_spectrum(self, z_target):
        spec_index = np.argmin(np.abs(self.z_s - z_target))
        return self.spectrum_list[spec_index]


class BrightnessCache:
    """
    Cache for the X-ray brightness boxes.

    Args:
        data_path (str): Path to the HDF5 cache file.

    Notes:
        Brightness = energy per averaged baryon.
    """

    def __init__(self, data_path):

        self.data_path = data_path
        self.redshifts = np.array([])

    def cache_box(self, box, redshift):
        """
        Adds the X-ray box box at the specified redshift to the cache.

        Args:
            box (np.ndarray): The X-ray brightness box to cache. (photons / Mpccm^3)
            redshift (float): The redshift of the box.
        """

        box_index = len(self.redshifts)

        with h5py.File(self.data_path, 'a') as archive:
            archive.create_dataset('Box_' + str(box_index), data = fft.rfftn(box))
            
        self.redshifts = np.append(self.redshifts, redshift)

    def get_box(self, redshift):
        """Returns the brightness box and spectrum at the specified cached state."""

	    # Get the index of the donor box
        box_index = np.argmin(np.abs(self.redshifts - redshift))

        with h5py.File(self.data_path, 'r') as archive:
            box = np.array(archive['Box_' + str(box_index)], dtype = complex)
        
        return box
