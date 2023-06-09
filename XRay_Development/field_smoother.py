import sys, os, h5py
import numpy as np
from scipy import signal, ndimage, stats, interpolate

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
        """
        Returns the proper null geodesic distance between z_receiver and {z1, z2} 
        in units of Mpc
        """

        R1 = self.cosmo.lookback_distance(z1) - self.cosmo.lookback_distance(z_receiver)
        R2 = self.cosmo.lookback_distance(z2) - self.cosmo.lookback_distance(z_receiver)

        return R1.value, R2.value

    def get_smoothed_shell(z_receiver, z_donor, z_next_donor):

        # Get the index of the donor field
        field_index = np.argmin(np.abs(self.redshifts - redshift))

        # Get the smoothing radii and canonically sort them
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

        # Geometric factors for calculating X-rays incident on a point
        shell_volume = 4*np.pi/ 3 * (R2**3 - R1**3) # volume of the comoving shell
        solid_angle = 1 / np.pi / (R2 + R1)**2 # the solid angle of the target 

        # Load the field and smooth via FFT
        field, spec = self.get_field(field_index)
        return shell_volume * solid_angle * np.fft.irfftn(field * W), spec
