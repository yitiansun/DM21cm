"""Modifiers for injection classes."""

import os
import sys

import numpy as np
from astropy.cosmology import Planck18 as cosmo

sys.path.append(os.environ['DM21CM_DIR'])
from dm21cm.injections.base import Injection


class Multiplier (Injection):
    """Injection wrapper for modifying injection rates based on redshift and state of the universe.

    Args:
        injection (Injection): Injection object.
        multiplier_at_z (callable): Multiplier function that takes redshift and state of the universe as input.
        signature (str, optional): Signature for the modifier. Defaults to empty string.
    """

    def __init__(self, injection, multiplier_at_z, signature=''):
        self.injection = injection
        self.multiplier_at_z = multiplier_at_z
        self.signature = signature

    def multiplier_step(self, z_start, z_end=None, state=None, n_sample_pt=1000):
        """Multiplier for injection rate for a step in redshift.
        
        Args:
            z_start (float): Starting redshift of the redshift step of injection.
            z_end (float, optional): Ending redshift of the redshift step of injection. See details in inj_rate.
            state (dict, optional): State of the universe at z_start. Used for rates with feedback.

        Returns:
            float: Multiplier for injection rate.
        """
        if z_end is None:
            m = self.multiplier_at_z(z_start, state=state)
        else:
            z_s = np.linspace(z_start, z_end, n_sample_pt) # descending
            t_s = cosmo.age(z_s).value # ascending
            multiplier_s = np.array([self.multiplier_at_z(z, state=state) for z in z_s])
            m = np.trapz(multiplier_s, t_s) / (t_s[-1] - t_s[0])
            
        if np.abs(m) < 1e-100:
            m = 1e-100 # avoid division by zero
        return m
    
    #===== Injection API =====
    # utilities
    def is_injecting_elec(self):
        return self.injection.is_injecting_elec()
    
    def get_config(self):
        config = self.injection.get_config()
        config['signature'] = self.signature
        return config
    
    def __eq__(self, other):
        """Equality comparison using self.get_config."""
        return self.get_config() == other.get_config()
    
    def __repr__(self):
        """Representation of the injection."""
        return f"{self.__class__.__name__}({self.injection.__class__.__name__})({self.get_config()})"

    # injections
    def inj_rate(self, z_start, z_end=None, state=None, **kwargs):
        m = self.multiplier_step(z_start, z_end=z_end, state=state)
        return m * self.injection.inj_rate(z_start, z_end=z_end, state=state, **kwargs)
    
    def inj_power(self, z_start, z_end=None, state=None, **kwargs):
        m = self.multiplier_step(z_start, z_end=z_end, state=state)
        return m * self.injection.inj_power(z_start, z_end=z_end, state=state, **kwargs)
    
    def inj_phot_spec(self, z_start, z_end=None, state=None, **kwargs):
        m = self.multiplier_step(z_start, z_end=z_end, state=state)
        return float(m) * self.injection.inj_phot_spec(z_start, z_end=z_end, state=state, **kwargs)
    
    def inj_elec_spec(self, z_start, z_end=None, state=None, **kwargs):
        m = self.multiplier_step(z_start, z_end=z_end, state=state)
        return float(m) * self.injection.inj_elec_spec(z_start, z_end=z_end, state=state, **kwargs)
    
    def inj_phot_spec_box(self, z_start, z_end=None, state=None, **kwargs):
        m = self.multiplier_step(z_start, z_end=z_end, state=state)
        spec, weight_box = self.injection.inj_phot_spec_box(z_start, z_end=z_end, state=state, **kwargs)
        return float(m) * spec, weight_box
    
    def inj_elec_spec_box(self, z_start, z_end=None, state=None, **kwargs):
        m = self.multiplier_step(z_start, z_end=z_end, state=state)
        spec, weight_box = self.injection.inj_elec_spec_box(z_start, z_end=z_end, state=state, **kwargs)
        return float(m) * spec, weight_box