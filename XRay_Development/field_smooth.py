import sys, os, h5py
import numpy as np
from scipy import signal, ndimage, stats, interpolate
import matplotlib as mpl
import matplotlib.pyplot as plt


# For now we are using a gaussian window. May make more sense to go to a top-hat window function later.
class WindowedData:
    
    def __init__(self, data_path, N=None, dx=None, field=None):
        
        self.data_path = data_path

        if field is not None:
            self.N = N
            self.dx = dx
            self.field = np.fft.rfftn(field)
        
            # The wavenumbers we need to do the smoothing
            k = np.fft.fftfreq(N, d = dx)
            kReal = np.fft.rfftfreq(N, d = dx)

            self.kMag = 2*np.pi*np.sqrt(k[:, None, None]**2 + k[None, :, None]**2 + kReal[None, None, :]**2)
        
    def smooth(self, R):
        # If R = 0, then we do no smoothing
        if R == 0:
            return np.fft.irfftn(self.field)
        
        window_function = np.exp(-(self.kMag*R)**2/2)
        return np.fft.irfftn(self.field * window_function)
    
    def make_smoothed_fields(self, r_list, clean = True):
        r_list = np.sort(r_list)
        
        with h5py.File(self.data_path, 'a') as archive:            
            archive.create_dataset('Smoothing_Radius', data=r_list)
            
            for i, r in enumerate(np.append([0], r_list)):
                archive.create_dataset('RSmoothing_' + str(i), data = self.smooth(r))
          
        # By default, we clear the class variables to save memory. This allows us to 
        # keep this class present for convenient access via the `access_data` method
        # but will prevent us from recomputing in the future. 
        if clean:
            self.N = None
            self.dx = None
            self.field = None
            self.kMag = None
            
    def access_data(self, r, recompute=False):
        with h5py.File(self.data_path, 'r') as archive:            
            r_list = np.array(archive['Smoothing_Radius'])

            # If we want to recompute, do that but only if the field is present
            if recompute and self.field != None:
                return smooth(r)
            
            # Otherwise, return precomputed data, by log-interpolation, 
            # handling edge cases.
            elif r == 0:
                return np.array(archive['RSmoothing_0'])
            elif r > r_list[-1]:
                return np.array(archive['RSmoothing_' + str(len(r_list)-1)])

            # These are the indices for the data we want to use in the interpolatino
            upper_index = np.searchsorted(r_list, r)
            lower_index = upper_index - 1

            # These are the smoothed boxes
            lower_data = archive['RSmoothing_' + str(lower_index)]
            high_data = archive['RSmoothing_' + str(upper_index)]

            # This defines the interpolation range and target
            interp_range = np.log(r_list[lower_index:upper_index+1])
            interp_target = np.log(r)
            
            # Perform the interpolation and return the result
            return interpolate.interp1d(interp_range, np.stack((lower_data, high_data)), axis = 0)(interp_target)


# This is a convenience wrapper
class SmootherArray:
    
    def __init__(self, ):
        
        self.redshifts = np.array([])
        self.smoothed_data = np.array([])
        
    def add_data(self, z, smoother):

        # Prepends the new data to have z-increasing ordering
        self.redshifts = np.append(z, self.redshifts)
        self.smoothed_data = np.append(smoother, self.smoothed_data)
        
    def access_data(self, z, r):
        
        # This is finding where to access the smoother
        upper_index = np.searchsorted(self.redshifts, z)
        lower_index = upper_index - 1
        
        # These are the smoothed boxes
        upper_data = self.smoothed_data[upper_index].access_data(r)
        lower_data = self.smoothed_data[lower_index].access_data(r)
        
        # This defines the interpolation range
        interp_range = self.redshifts[lower_index:upper_index+1]

        # Perform the interpolation and return the result
        return interpolate.interp1d(interp_range, np.stack((lower_data, upper_data)), axis = 0)(z)
        
        
