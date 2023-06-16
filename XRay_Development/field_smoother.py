import sys, os, h5py
import numpy as np
from scipy import signal, ndimage, stats, interpolate, integrate
from astropy import cosmology, constants, units

class WindowedData:
    def __init__(self, data_path, cosmo, N, dx, cache=True, irfftn=np.fft.irfftn):
        """
        Class initializer. The arguments are:
        'data_path' - the path where the caching hdf5 is stored
        'cosmo'     - an instance of an astropy cosmology
        'N'         - the HII_DIM from 21cmFAST
        'dx'        - the pixel sidelength for 21cmFAST data cubes in Mpccm
        'cache'     - Boolean controlling if data is cached (True) or kept in memory (False)
        'irfftn'    - Method for performing the 3-D inverse real valued fft
        """

        self.data_path = data_path
        self.cosmo = cosmo
        self.redshifts = np.array([])
        self.irfftn = irfftn
        self.cache = cache

        if not self.cache:
            self.boxes = []
            self.specs = []

        # Generate the kmagnitudes and save them
        k = np.fft.fftfreq(N, d = dx)
        kReal = np.fft.rfftfreq(N, d = dx)
        self.kMag = 2*np.pi*np.sqrt(k[:, None, None]**2 + k[None, :, None]**2 + kReal[None, None, :]**2)

        # Arrays for tracking the global quantities
        self.global_Tk = np.zeros((0))
        self.global_x = np.zeros((0))

    def purge(self, zmax):
        """
        This method clears unnecessary data from the in-memory list if caching is not being used.
        'zmax' - data from redshifts above this are purged
        """
        if self.cache:
            return None

        for i, z in enumerate(self.redshifts):
            if z > zmax:
                self.specs[i] = None
                self.boxes[i] = None

    def set_field(self, field, spec, z):
        """
        This method adds the X-ray brightness field and the X-ray spectrum at a specified
        redshift to the cache.
        'field' - the comoving volumetric emissivity of photons in each pixel. Units of photonos / Mpccm^3
        'spec' - the X-ray spectrum, can be in whatever units you want, this is just for caching convenience
        'z'    - the redshift associated with the brightness field and spectrum
        """
        self.redshifts = np.append(self.redshifts, z)

        if self.cache:
            field_index = len(self.redshifts)

            with h5py.File(self.data_path, 'a') as archive:
                archive.create_dataset('Field_' + str(field_index), data = np.fft.rfftn(field))
                archive.create_dataset('Spec_' + str(field_index), data = spec)

        else:
            self.boxes.append(field)
            self.specs.append(spec)


    def _get_field(self, field_index):
        """
        Returns the brightness field and spectrum at the specified cached state. 
        """

        if self.cache:
            with h5py.File(self.data_path, 'r') as archive:
                field = np.array(archive['Field_' + str(field_index)], dtype = complex)
                spec = np.array(archive['Spec_' + str(field_index)], dtype = float)

        else:
            field = self.field[field_index]
            spec = self.specs[field_index]

        return field, spec

    def _get_smoothing_radii(self, z_receiver, z1, z2):
        """
        Evaluates the shell radii for a receiver at `z_receiver` for emission between redshifts
        `z1` and `z2`
        """

        z_at_t = lambda t: cosmology.z_at_value(self.cosmo.age, t)
        integrand = lambda t: 1+z_at_t(t*units.Gyr)

        t_receiver = self.cosmo.age(z_receiver).value
        t1 = self.cosmo.age(z1).value
        t2 = self.cosmo.age(z2).value

        R1 = integrate.quad(integrand, t_receiver, t1)[0] *constants.c.to(units.Mpc / units.Gyr)*units.Gyr
        R2 = integrate.quad(integrand, t_receiver, t2)[0] *constants.c.to(units.Mpc / units.Gyr)*units.Gyr

        return np.abs(R1.value), np.abs(R2.value)

    def get_smoothed_shell(self, z_receiver=None, z_donor=None, z_next_donor=None):
        '''
        Calculate the spatially-dependent intensity of X-rays photons at `z_receiver`
        for photons emitted as early as `z_donor` and as late as `z_next_donor`.

        Returns the incident intensity in photons/Mpc^2 and the spectrum (without redshifting)
        '''

        # Get the index of the donor field
        field_index = np.argmin(np.abs(self.redshifts - z_donor))

        # Get the smoothing radii in comoving coordinates and canonically sort them
        R1, R2 = self._get_smoothing_radii(z_receiver, z_donor, z_next_donor)

        # Volumetric weighting factors for combining the window functions
        R1, R2 = np.sort([R1, R2])
        w1 = R1**3 / (R2**3 - R1**3)
        w2 = R2**3 / (R2**3 - R1**3)

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
        
        # Load the field and smooth via FFT
        field, spec = self._get_field(field_index)
        field = self.irfftn(field * W) # This is an averaging
        
        return field, spec
        # Fix what is below later.
        # Comoving shell volume
        shell_volume = 4*np.pi/ 3 * (R2**3 - R1**3) # volume of the comoving shell

        # Target solid angle
        Rp = self.cosmo.lookback_distance(z_donor) - self.cosmo.lookback_distance(z_receiver)
        solid_angle = 1 / 4 / np.pi /Rp.value**2

        
        return shell_volume * solid_angle * field, spec
