import sys, os, h5py
import numpy as np
from scipy import signal, ndimage, stats, interpolate, integrate
from astropy import cosmology, constants, units

class WindowedData:
    def __init__(self, data_path, cosmo, N, dx, N_x):

        self.data_path = data_path
        self.cosmo = cosmo
        self.redshifts = np.array([])

        # Generate the kmagnitudes and save them
        k = np.fft.fftfreq(N, d = dx)
        kReal = np.fft.rfftfreq(N, d = dx)
        self.kMag = 2*np.pi*np.sqrt(k[:, None, None]**2 + k[None, :, None]**2 + kReal[None, None, :]**2)

        # Arrays for tracking the global quantities
        self.global_Tk = np.zeros((0))
        self.global_x = np.zeros((0))

    def set_field(self, field, spec, z):
        """
        Adds the provided X-Ray brightness field and spectrum to the hdf5 cache
        """

        field_index = len(self.redshifts)

        with h5py.File(self.data_path, 'a') as archive:
            archive.create_dataset('Field_' + str(field_index), data = np.fft.rfftn(field))
            archive.create_dataset('Spec_' + str(field_index), data = spec)

        self.redshifts = np.append(self.redshifts, z)

    def get_field(self, field_index):

        with h5py.File(self.data_path, 'r') as archive:
            return np.array(archive['Field_' + str(field_index)], dtype = complex), np.array(archive['Spec_' + str(field_index)], dtype = float)

    def get_smoothing_radii(self, z_receiver, z1, z2):
        z_at_t = lambda t: cosmology.z_at_value(self.cosmo.age, t)
        integrand = lambda t: 1+z_at_t(t*units.Gyr)

        t_receiver = self.cosmo.age(z_receiver).value
        t1 = self.cosmo.age(z1).value
        t2 = self.cosmo.age(z2).value

        R1 = integrate.quad(integrand, t_receiver, t1)[0] *constants.c.to(units.Mpc / units.Gyr)*units.Gyr
        R2 = integrate.quad(integrand, t_receiver, t2)[0] *constants.c.to(units.Mpc / units.Gyr)*units.Gyr

        return np.abs(R1.value), np.abs(R2.value)

    def get_smoothed_shell(z_receiver, z_donor, z_next_donor):
        '''
        Smooths the shells for the given specification of the receiver
        redshift `z_receiver`, from a shell with outer radius defined by
        `z_donor` and an inner radius defined by `z_next_donor`.
        '''

        # Get the index of the donor field
        field_index = np.argmin(np.abs(self.redshifts - redshift))

        # Get the smoothing radii in comoving coordinates and canonically sort them
        R1, R2 = get_smoothing_radii(self, z_receiver, z_donor, z_next_donor)

        # Volumetric weighting factors for combining the window functions
        R1, R2 = np.sort([R1, R2])
        w1 = R1**3 / (R2**3 - R1**3)
        w2 = R2**3 / (R2**3 - R1**3)

        # Construct the smoothing functions in the frequency domain
        W1 = 3*(np.sin(self.kMag*R1) - self.kMag*R1 * np.cos(self.kMag*R1)) /(self.kMag*R1)**3
        W2 = 3*(np.sin(self.kMag*R2) - self.kMag*R2 * np.cos(self.kMag*R2)) /(self.kMag*R2)**3

        # Fix the nan issue
        W1[0, 0, 0] = 1
        W2[0, 0, 0] = 1

        # Combine the window functions
        W = w2*W2 - w1*W1
        del W1, W2

        # Comoving shell volume
        shell_volume = 4*np.pi/ 3 * (R2**3 - R1**3) # volume of the comoving shell

        # Target solid angle
        Rp = self.cosmo.lookback_distance(z_donor) - self.cosmo.lookback_distance(z_receiver)
        solid_angle = 1 / 4 / np.pi /Rp.value**2

        # Load the field and smooth via FFT
        field, spec = self.get_field(field_index)
        return shell_volume * solid_angle * np.fft.irfftn(field * W), spec

