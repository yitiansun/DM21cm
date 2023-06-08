import sys, os, h5py
import numpy as np
from scipy import signal, ndimage, stats, interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt

class WindowedData:
    def __init__(self, data_path, N, dx, N_x, flush_k = False):
         
        self.data_path = data_path
        self.redshifts = np.array([])
        
        # Generate the kmagnitudes and save them
        k = np.fft.fftfreq(N, d = dx)
        kReal = np.fft.rfftfreq(N, d = dx)
        self.kMag = 2*np.pi*np.sqrt(k[:, None, None]**2 + k[None, :, None]**2 + kReal[None, None, :]**2)
        
        with h5py.File(self.data_path, 'a') as archive: 
            archive.create_dataset('kMag', data=self.kMag)
            
        self.global_Tk = np.zeros((0))
        self.global_x = np.zeros((0))
        self.global_tau = np.zeros((0, N_x))
    
    def set_field(self, field, spec, z):
        field_index = len(self.redshifts)
        
        with h5py.File(self.data_path, 'a') as archive:
            archive.create_dataset('Field_' + str(field_index), data = np.fft.rfftn(field))
            archive.create_dataset('Spec_' + str(field_index), data = spec)
        
        self.redshifts = np.append(self.redshifts, z)
        
    def get_field(self, field_index):
        with h5py.File(self.data_path, 'r') as archive:
            return np.array(archive['Field_' + str(field_index)], dtype = complex), np.array(archive['Spec_' + str(field_index)], dtype = float)


    def smoothed_shell(self, redshift, R1, R2):
        # Get the index of the nearest field in redshift
        field_index = np.argmin(np.abs(self.redshifts - redshift))

        # Canonically sort the radii
        R1, R2 = np.sort([R1, R2])
        w1 = R1**3 / (R2**3 - R1**3)
        w2 = R2**3 / (R2**3 - R1**3)

        # Construct the smoothing function 
        W1 = 3*(np.sin(self.kMag*R1) - self.kMag*R1 * np.cos(self.kMag*R1)) /(self.kMag*R1)**3 
        W2 = 3*(np.sin(self.kMag*R2) - self.kMag*R2 * np.cos(self.kMag*R2)) /(self.kMag*R2)**3 
        
        # Fix the nan issue
        W1[0, 0, 0] = 1
        W2[0, 0, 0] = 1

        # Combine the window functions
        W = w2*W2 - w1*W1
        del W1, W2

        # Load the field and define the smoothing functions
        field, spec = self.get_field(field_index)
        return np.fft.irfftn(field * W), spec
