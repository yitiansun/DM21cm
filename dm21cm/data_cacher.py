import sys, os, h5py
import numpy as np
from scipy import signal, ndimage, stats, interpolate, integrate
from astropy import cosmology, constants, units

class SpectrumCache:
    
    def __init__(self):
        self.spectrum_list = []
        self.redshift_list = []
        
    def append(self, spec, z):
        self.spectrum_list.append(spec)
        self.redshift_list.append(z)
        
    def attenuate(self, attenuation_factor):
        for spec in self.spectrum_list:
            spec.N *= attenuation_factor
            
    def redshift(self, target_z):
        for spec in self.spectrum_list:
            spec.redshift(1+target_z)
            
    def get_spectrum(self, target_z):
        spec_index = np.argmin(np.abs(np.array(self.redshift_list) - target_z))
        return self.spectrum_list[spec_index]


class BrightnessCache:
    def __init__(self, data_path, cosmo, N, dx):
        """
        Class initializer. The arguments are:
        'data_path' - the path where the caching hdf5 is stored
        'cosmo'     - an instance of an astropy cosmology
        'N'         - the HII_DIM from 21cmFAST
        'dx'        - the pixel sidelength for 21cmFAST data cubes in Mpccm
        'irfftn'    - Method for performing the 3-D inverse real valued fft
        """

        self.data_path = data_path
        self.cosmo = cosmo
        self.redshifts = np.array([])
        self.dx = dx
        self.N = N

        # Generate the kmagnitudes and save them
        k = np.fft.fftfreq(N, d = dx)
        kReal = np.fft.rfftfreq(N, d = dx)
        self.kMag = 2*np.pi*np.sqrt(k[:, None, None]**2 + k[None, :, None]**2 + kReal[None, None, :]**2)

        # Arrays for tracking the global quantities
        self.global_Tk = np.zeros((0))
        self.global_x = np.zeros((0))

    def cache_box(self, box, z):
        """
        This method adds the X-ray box box at the specified redshift to the cache
        'box' - the comoving effective density of photons in each pixel. Units of photons / Mpccm^3
        'z'    - the redshift associated with the brightness box
        """

        box_index = len(self.redshifts)

        with h5py.File(self.data_path, 'a') as archive:
            archive.create_dataset('Box_' + str(box_index), data = np.fft.rfftn(box))
            
        self.redshifts = np.append(self.redshifts, z)

    def _get_box(self, box_index):
        """
        Returns the brightness box and spectrum at the specified cached state. 
        """

        with h5py.File(self.data_path, 'r') as archive:
            box = np.array(archive['Box_' + str(box_index)], dtype = complex)
        return box
    
    
    def z_at_t(self, t):
        return cosmology.z_at_value(self.cosmo.age, t*units.Gyr)
    
    def rC_Integrand(self, t):
        return 1+self.z_at_t(t)
        
    
    def get_smoothing_radii(self, z_receiver, z1, z2):
        """
        Evaluates the shell radii for a receiver at `z_receiver` for emission between redshifts
        `z1` and `z2`
        """

        t_receiver = self.cosmo.age(z_receiver).value
        t1 = self.cosmo.age(z1).value
        t2 = self.cosmo.age(z2).value

        # Comoving separations
        R1 = integrate.quad(self.rC_Integrand, t_receiver, t1)[0] * units.Gyr * constants.c
        R2 = integrate.quad(self.rC_Integrand, t_receiver, t2)[0] * units.Gyr * constants.c
        
        # Need R1 and R2 in Comoving Mpc for the smoothing operation
        R1_Mpc = np.abs(R1.to('Mpc').value)
        R2_Mpc = np.abs(R2.to('Mpc').value)
        
        return R1_Mpc, R2_Mpc
     
    def get_smoothed_shell(self, z_receiver, z_donor, z_next_donor):
        '''
        Calculate the spatially-dependent intensity of X-rays photons at `z_receiver`
        for photons emitted as early as `z_donor` and as late as `z_next_donor`.
        
        Returns the instantaneous photon density in units of photons / Mpccm^3 for photons emitted
        between `z_donor` and `z_next_donor` arriving at `z_receiver`. Also returns the spectrum.
        '''

        # Get the index of the donor box
        box_index = np.argmin(np.abs(self.redshifts - z_donor))
        a_receiver=1. / (1+z_receiver)

        # Get the smoothing radii in comoving coordinates and canonically sort them
        R1, R2 = self.get_smoothing_radii(z_receiver, z_donor, z_next_donor)
        print(z_receiver, z_donor, z_next_donor, R1, R2)

        # Skip the smoothing if we are smoothing on a very large radius
        if min(R1,R2) > self.N//2*self.dx:
            box = self._get_box(box_index)
            box = np.fft.irfftn(box)
            box = np.mean(box)*np.ones_like(box)
            return box, True

        # Volumetric weighting factors for combining the window functions
        R1, R2 = np.sort([R1, R2])
        w1 = R1**3 / (R2**3 - R1**3)
        w2 = R2**3 / (R2**3 - R1**3)
        R1 = np.clip(R1, 1e-6, None)
        R2 = np.clip(R2, 1e-6, None)

        # Construct the smoothing functions in the frequency domain
        with np.errstate(divide='ignore'):
            W1 = 3*(np.sin(self.kMag*R1) - self.kMag*R1 * np.cos(self.kMag*R1)) /(self.kMag*R1)**3
            W2 = 3*(np.sin(self.kMag*R2) - self.kMag*R2 * np.cos(self.kMag*R2)) /(self.kMag*R2)**3

            # Fix the nan issue
            W1[0, 0, 0] = 1
            W2[0, 0, 0] = 1

        # Combine the window functions
        W = w2*W2 - w1*W1
        del W1, W2
        
        # Load the box and smooth via FFT
        box = self._get_box(box_index)
        
        # Multiply by flux_factor. `box` is now the equivalent universe density of photons.
        box = np.fft.irfftn(box * W) 
        
        return box, False
