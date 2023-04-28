import sys, os, h5py
import numpy as np
from scipy import signal, ndimage, stats, interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt

class WindowedData:
    def __init__(self, data_path, N, dx):
         
        self.data_path = data_path
        self.redshifts = np.array([])
        
        # Generate the kmagnitudes and save them
        k = np.fft.fftfreq(N, d = dx)
        kReal = np.fft.rfftfreq(N, d = dx)
        self.kMag = 2*np.pi*np.sqrt(k[:, None, None]**2 + k[None, :, None]**2 + kReal[None, None, :]**2)
        
        with h5py.File(self.data_path, 'a') as archive: 
            archive.create_dataset('kMag', data=self.kMag)
    
    def set_field(self, field, z):
        field_index = len(self.redshifts)
        
        with h5py.File(self.data_path, 'a') as archive:
            archive.create_dataset('Field_' + str(field_index), data = np.fft.rfftn(field))
        
        self.redshifts = np.append(self.redshifts, z)
        
    def get_field(self, field_index):
        with h5py.File(self.data_path, 'r') as archive:
            return np.array(archive['Field_' + str(field_index)], dtype = complex)
        
    def smoothed_shell(self, redshift, R1, R2):
        # Get the index of the nearest field in redshift
        field_index = np.argmin(np.abs(self.redshifts - redshift))
        
        # Canonically sort the radii
        R1, R2 = np.sort([R1, R2])

        # Load the field and define the smoothing functions
        field = self.get_field(field_index)
        W = np.exp(-(self.kMag*R2)**2/2) - (R1/R2)**3*np.exp(-(self.kMag*R1)**2/2)
        
        return np.fft.irfftn(field * W)
