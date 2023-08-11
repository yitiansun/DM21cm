"""Some utilities for the whole project"""

import h5py
import numpy as np
import py21cmfast as p21c

def get_z_edges(zmax, min_redshift, z_step_factor):
    redshifts = [min_redshift]
    while redshifts[-1] < zmax:
        redshifts.append((redshifts[-1] + 1.0) * z_step_factor - 1.0)

    return np.clip(redshifts[::-1], None, zmax)

def split_xray(phot_N, ix_lo, ix_hi):
    """Split a photon spectrum (N in bin) into bath and xray band."""
    bath_N = np.array(phot_N).copy()
    xray_N = np.array(phot_N).copy()
    bath_N[ix_lo:ix_hi] *= 0
    xray_N[:ix_lo] *= 0
    xray_N[ix_hi:] *= 0
    
    return bath_N, xray_N

def gen_injection_boxes(next_z, p21c_initial_conditions):
    
    # Instantiate the injection arrays
    input_heating = p21c.input_heating(redshift=next_z, init_boxes=p21c_initial_conditions, write=False)
    input_ionization = p21c.input_ionization(redshift=next_z, init_boxes=p21c_initial_conditions, write=False)
    input_jalpha = p21c.input_jalpha(redshift=next_z, init_boxes=p21c_initial_conditions, write=False)
    
    return input_heating, input_ionization, input_jalpha

def p21_step(z_eval, perturbed_field, spin_temp, ionized_box,
             input_heating = None, input_ionization = None, input_jalpha = None):
    
    # Calculate the spin temperature, possibly using our inputs
    spin_temp = p21c.spin_temperature(perturbed_field=perturbed_field,
                                      previous_spin_temp = spin_temp,
                                      input_heating_box = input_heating,
                                      input_ionization_box = input_ionization,
                                      input_jalpha_box = input_jalpha, )
    
    # Calculate the ionized box
    ionized_box = p21c.ionize_box(perturbed_field = perturbed_field,
                                  previous_ionize_box=ionized_box,
                                  spin_temp=spin_temp)
    
    
    # Calculate the brightness temperature
    brightness_temp = p21c.brightness_temperature(ionized_box=ionized_box,
                                                  perturbed_field = perturbed_field,
                                                  spin_temp = spin_temp)
    
    # Now return the spin temperature and ionized box because we will need them later
    return spin_temp, ionized_box, brightness_temp
