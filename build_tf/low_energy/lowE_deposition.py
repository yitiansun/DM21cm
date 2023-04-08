""" Modified from DarkHistory

    Total energy deposition from low-energy electrons and photons into the IGM.
  
    As detailed in Section III.F of the paper sub-3keV photons and electrons (dubbed low-energy photons and electrons) deposit their energy into the IGM in the form of hydrogen/helium ionization, hydrogen excitation, heat, or continuum photons, each of which corresponds to a channel, c.
"""

import os, sys
sys.path.append(os.environ['DH_DIR'])
sys.path.append('..')

from low_energy import lowE_electrons
from low_energy import lowE_photons

import numpy as np
import time

from darkhistory import physics as phys
from darkhistory.spec.spectrum import Spectrum
from darkhistory.spec import spectools


def compute_fs(MEDEA_interp, rs, x, elec_spec, phot_spec, dE_dVdt_inj, dt, highengdep, cmbloss=0, method='no_He', cross_check=False, ion_old=False, print_time=False):
    """ Compute f(z) fractions for continuum photons, photoexcitation of HI, and photoionization of HI, HeI, HeII

    Given a spectrum of deposited electrons and photons, resolve their energy into
    H ionization, and ionization, H excitation, heating, and continuum photons in that order.

    Parameters
     ----------
    phot_spec : Spectrum object
        spectrum of photons. Assumed to be in dNdE mode. spec.totN() should return number *per baryon*.
    elec_spec : Spectrum object
        spectrum of electrons. Assumed to be in dNdE mode. spec.totN() should return number *per baryon*.
    x : list of floats
        number of (HI, HeI, HeII) divided by nH at redshift photon_spectrum.rs
    dE_dVdt_inj : float
        DM energy injection rate, dE/dVdt injected.  This is for unclustered DM (i.e. without structure formation).
    dt : float
        time in seconds over which these spectra were deposited.
    highengdep : list of floats
        total amount of energy deposited by high energy particles into {H_ionization, H_excitation, heating, continuum} per baryon per time, in that order.
    cmbloss : float
        Total amount of energy in upscattered photons that came from the CMB, per baryon per time, (1/n_B)dE/dVdt. Default is zero.
    method : {'no_He', 'He_recomb', 'He'}
        Method for evaluating helium ionization. 

        * *'no_He'* -- all ionization assigned to hydrogen;
        * *'He_recomb'* -- all photoionized helium atoms recombine; and 
        * *'He'* -- all photoionized helium atoms do not recombine.

    Returns
    -------
    ndarray or tuple of ndarray
    f_c(z) for z within spec.rs +/- dt/2
    The order of the channels is {H Ionization, He Ionization, H Excitation, Heating and Continuum} 

    Notes
    -----
    The CMB component hasn't been subtracted from the continuum photons yet
    Think about the exceptions that should be thrown (elec_spec.rs should equal phot_spec.rs)
    
    Assuming average nB!!! nBscale not needed. Use the average nB to compute dE_dVdt_inj!!
    """

    # np.array syntax below needed so that a fresh copy of eng and N are passed to the
    # constructor, instead of simply a reference.
    
    if method == 'no_He':
        
        if print_time:
            timer = time.time()
        
        if ion_old:
            ion_bounds = spectools.get_bounds_between(
                phot_spec.eng, phys.rydberg
            )
            ion_engs = np.exp((np.log(ion_bounds[1:])+np.log(ion_bounds[:-1]))/2)

            ionized_elec = Spectrum(
                ion_engs,
                phot_spec.totN(bound_type="eng", bound_arr=ion_bounds),
                rs=rs,
                spec_type='N'
            )
            new_eng = ion_engs - phys.rydberg
            ionized_elec.shift_eng(new_eng)
            
        else:
            i = np.searchsorted(phot_spec.eng, phys.rydberg)
            ionized_elec = Spectrum(
                phot_spec.eng[i:]-phys.rydberg,
                phot_spec.N[i:],
                rs=rs,
                spec_type='N'
            )
            
        ionized_elec.rebin(elec_spec.eng)
        sum_elec_spec = elec_spec + ionized_elec
        
        if print_time:
            print(f'T1 = {time.time()-timer:.6f}', end=' ')
            timer = time.time()
        
        f_phot = lowE_photons.compute_fs(
            phot_spec, x, dE_dVdt_inj, dt, method='old', cross_check=cross_check
        )
        
        if print_time:
            print(f'T2 = {time.time()-timer:.6f}', end=' ')
            timer = time.time()

        f_elec = lowE_electrons.compute_fs(
            MEDEA_interp, sum_elec_spec, 1-x[0], dE_dVdt_inj, dt
        )
        
        if print_time:
            print(f'T3 = {time.time()-timer:.6f}')
            timer = time.time()
        
        # f_low is {H ion, He ion, Lya Excitation, Heating, Continuum}
        f_low = np.array([
            f_phot[2]+f_elec[2],
            f_phot[3]+f_phot[4]+f_elec[3],
            f_phot[1]+f_elec[1],
            f_elec[4],
            f_phot[0]+f_elec[0] - cmbloss * phys.nB * rs**3 / dE_dVdt_inj
        ])

        f_high = np.array([
            highengdep[0],
            0,
            highengdep[1],
            highengdep[2],
            highengdep[3]
        ]) * phys.nB * rs**3 / dE_dVdt_inj

        return f_low, f_high

    else: 
        raise NotImplementedError