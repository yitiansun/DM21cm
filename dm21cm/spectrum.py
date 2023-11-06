import os
import sys
import numpy as np

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.spectrum import Spectrum


class AttenuatedSpectrum:

    def __init__(self, spectrum):
        self.spectrum = spectrum
        self.attenuation_arr = np.ones_like(spectrum.N)

    def attenuate(self, attenuation_arr):
        self.attenuation_arr *= attenuation_arr

    def redshift(self, rs): # this is approximate
        self.spectrum.redshift(rs)

    def switch_spec_type(self, spec_type):
        self.spectrum.switch_spec_type(spec_type)

    @property
    def approx_attenuation_arr(self):
        return np.where(self.attenuation_arr > np.exp(-1), 1., 0.)
    
    @property
    def approx_attentuation_arr_repr(self):
        return ''.join([f'{a:.0f}' for a in self.approx_attenuation_arr])

    @property
    def approx_attenuated_spectrum(self):
        spec_N = self.spectrum.N * self.approx_attenuation_arr
        return Spectrum(self.spectrum.eng, spec_N, spec_type='N', rs=self.spectrum.rs)
    
    @property
    def attenuated_spectrum(self):
        spec_N = self.spectrum.N * self.attenuation_arr
        return Spectrum(self.spectrum.eng, spec_N, spec_type='N', rs=self.spectrum.rs)