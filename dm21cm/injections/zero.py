"""Zero injection (used by DarkHistoryWrapper)."""

import os
import sys

import numpy as np
import jax.numpy as jnp

sys.path.append(os.environ['DM21CM_DIR'])
import dm21cm.physics as phys
from dm21cm.injections.base import Injection

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.spectrum import Spectrum


class ZeroInjection (Injection):
    """Zero injection."""

    def __init__(self):
        self.mode = 'Zero injection'

    def set_binning(self, abscs):
        self.zero_phot_spec = Spectrum(abscs['photE'], 0. * abscs['photE'], spec_type='N')
        self.zero_elec_spec = Spectrum(abscs['elecEk'], 0. * abscs['elecEk'], spec_type='N')

    def is_injecting_elec(self):
        return False
    
    def get_config(self):
        return {
            'mode': self.mode
        }

    #===== injections =====
    def inj_rate(self, z_start, z_end=None, **kwargs):
        return 1e-100
    
    def inj_power(self, z_start, z_end=None, **kwargs):
        return 1e-100
    
    def inj_phot_spec(self, z_start, z_end=None, **kwargs):
        return self.zero_phot_spec
    
    def inj_elec_spec(self, z_start, z_end=None, **kwargs):
        return self.zero_elec_spec
    
    def inj_phot_spec_box(self, z_start, z_end=None, delta_plus_one_box=None, **kwargs):
        return self.zero_phot_spec, delta_plus_one_box

    def inj_elec_spec_box(self, z_start, z_end=None, delta_plus_one_box=None, **kwargs):
        return self.zero_elec_spec, delta_plus_one_box