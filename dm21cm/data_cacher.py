"""Xray data cacher classes."""

import os
import sys
import numpy as np

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
        z_start (float): The starting redshift of the step that generates the emissivity data.
        z_end (float): The redshift at the end of the step, at which the spectrum is saved.
        spectrum (Spectrum): The X-ray spectrum.
        box (array): The X-ray brightness box.
    """

    def __init__(self, z_start, z_end, spectrum, box):
        self.z_start = z_start
        self.z_end = z_end
        self.spectrum = spectrum
        self.spectrum.switch_spec_type('N')
        self.box = box

        self.ftbox = fft.rfftn(box)

    @property
    def dz(self):
        return self.z_start - self.z_end
    
    def attenuate(self, attenuation_arr):
        self.spectrum.N *= attenuation_arr

    def redshift(self, z_target):
        self.spectrum.redshift(1+z_target)
    

class Cacher:
    """Cacher for xray states.
    
    Args:
        box_dim (int): The dimension of the box.
        dx (float): The size of each cell [cfMpc].
    """

    def __init__(self, box_dim, dx, abscissa = np.zeros((500))):
        self.box_dim = box_dim    # The number of lattice sites in resolving the box
        self.dx = dx              # The linear dimension of the box, in comoving Mpc
        self.states = []          # A list containing our cached states
        self.abscissa = abscissa  # The bin abcissa for the spectrum

        # Generate the k-magnitudes for use in spectral smoothing operations
        k = fft.fftfreq(self.box_dim, d=self.dx)
        kReal = fft.rfftfreq(self.box_dim, d=self.dx)
        self.kMag = 2*jnp.pi*jnp.sqrt(k[:, None, None]**2 + k[None, :, None]**2 + kReal[None, None, :]**2)

    def cache(self, z_start, z_end, spectrum, box):
        self.states.append(CachedState(z_start, z_end, spectrum, box))
        
    @property
    def z_s(self):
        return np.array([state.z_end for state in self.states])

    def get_prior_state(self, z):
        prior_locs = np.where(self.zs > z)[0]
        if len(prior_locs) == 0:
            raise ValueError(f'z={z} out of bounds {np.min(z_s)} - {np.max(z_s)}.')

        prior_state = self.states[prior_locs][np.argmin(self.zs[prior_locs])]
        return prior_state

    def get_later_state(self, z):
        later_locs = np.where(self.zs < z)[0]
        if len(later_locs) == 0:
            
            # Make a null state and return it 
            null_spec = self.zeros_like(self.abscissa)
            null_box = np.zeros((self.box_dim, self.box_dim, self.box_dim))
            null_state = CachedState(z, z, null_spec, null_box)
            return null_state
        
        later_state= self.states[later_locs][np.argmax(self.zs[later_locs])]
        return later_state
    
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
        self.kMag = jnp.clip(self.kMag, EPSILON, None)

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

        # Perform the fourier transform and 
        return fft.irfftn(ftbox * W)
    
    def get_ftdEdz_spec(self, z):
        """Return the Fourier transform of dE/dz and the spectrum at the specified redshift."""
        state = self.get_state(z)
        mean_eng = state.spectrum.toteng()
        return state.ftbox * mean_eng / state.dz, state.spectrum / mean_eng
    
    def release_to_bath_prior_to(self, z):
        """
        Release the cached data prior z into the bath.
        """
        to_bath_locs = np.sort(np.where(self.z_s > z)[0])[::-1]
        to_bath_spectrum = np.zeros_like(self.abscissa)
        
        for index in to_bath_locs:
            to_bath_spectrum += self.states[index].spectrum
            del self.states[index]

        return to_bath_spectrum
