import sys, os, h5py
import numpy as np
from scipy import signal, ndimage, stats, interpolate

class WindowedData:
    def __init__(self, data_path, N, L):

        self.data_path = data_path
        self.zs = np.array([])

        # Generate the kmagnitudes and save them
        dx = N / L
        k = np.fft.fftfreq(N, d = dx)
        kReal = np.fft.rfftfreq(N, d = dx)
        self.kMag = 2*np.pi*np.sqrt(k[:, None, None]**2 + k[None, :, None]**2 + kReal[None, None, :]**2)

        with h5py.File(self.data_path, 'a') as archive:
            archive.create_dataset('kMag', data=self.kMag)

    def set_field(self, field, spec, z):
        z_index = len(self.zs)

        with h5py.File(self.data_path, 'a') as archive:
            archive.create_dataset('Field_' + str(z_index), data = np.fft.rfftn(field / np.mean(field)))
            archive.create_dataset('Spectrum_' + str(z_index), data = spec)

        self.zs = np.append(self.zs, z_index)

    def get_field(self, z_index):
        with h5py.File(self.data_path, 'r') as archive:
            return np.array(archive['Field_' + str(z_index)], dtype = complex)

    def get_smoothed_shell(self, z, R1, R2):
        # Get the index of the nearest field in redshift
        z_index = np.argmin(np.abs(self.zs - z))

        # Canonically sort the radii
        R1, R2 = np.sort([R1, R2])

        # Load the field and define the smoothing functions
        field = self.get_field(z_index)
        W = np.exp(-(self.kMag*R2)**2/2) - (R1/R2)**3*np.exp(-(self.kMag*R1)**2/2)

        return np.fft.irfftn(field * W)

    def get_spectrum(self, z):
        z_index = np.argmin(np.abs(self.zs - z))

        with h5py.File(self.data_path, 'a') as archive:
            return np.array(archive['Spectrum_' + str(z_index)])
