"""Physics constants and functions."""

import os
import sys
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy import constants as const
from scipy import interpolate
from scipy import integrate

sys.path.append(os.environ['DH_DIR'])
from darkhistory.config import load_data as dh_load_data


#===== Units in cgs =====
Mpc = 3.08568e24
"""Mpc in cm."""


#===== Fundamental Particles and Constants =====
# (Based on darkhistory.physics)

m_p          = 0.938272081e9
"""Proton mass in eV."""
m_e          = 510998.9461
"""Electron mass in eV."""
m_He         = 3.97107*m_p
"""Helium nucleus mass in eV."""
hbar         = 6.58211951e-16
"""hbar in eV s."""
c            = 299792458e2
"""Speed of light in cm s\ :sup:`-1`\ ."""
kB           = 8.6173324e-5
"""Boltzmann constant in eV K\ :sup:`-1`\ ."""
rydberg      = 13.60569253
"""Ionization potential of ground state hydrogen in eV."""
lya_eng      = rydberg*3/4
"""Lyman alpha transition energy in eV."""
He_ion_eng   = 24.587387936
"""First ionization energy of helium in eV."""


#===== Densities and Hubble =====
# (Based on darkhistory.physics)

h    = cosmo.h
""" h parameter."""
H0   = cosmo.H0.to('1/s').value
""" Hubble parameter today in s\ :sup:`-1`\ ."""
omega_m      = cosmo.Om0
""" Omega of all matter today."""
omega_rad    = cosmo.Ogamma0
""" Omega of radiation today."""
omega_lambda = cosmo.Ode0
""" Omega of dark energy today."""
omega_baryon = cosmo.Ob0
""" Omega of baryons today."""
omega_DM     = cosmo.Odm0
""" Omega of dark matter today."""
rho_crit     = (cosmo.critical_density0 * (const.c)**2).to('eV/cm^3').value
""" Critical density of the universe in eV cm\ :sup:`-3`\ ."""
rho_DM       = rho_crit*omega_DM
""" DM density in eV cm\ :sup:`-3`\ ."""
rho_baryon   = rho_crit*omega_baryon
""" Baryon density in eV cm\ :sup:`-3`\ ."""
n_B          = rho_baryon/m_p
""" Baryon number density in cm\ :sup:`-3`\ ."""

Y_He         = 0.245
"""Helium abundance by mass."""
n_H          = (1-Y_He) * n_B
"""Hydrogen nucleus number density in cm\ :sup:`-3`\ ."""
n_He         = (Y_He/4) * n_B
"""Helium nucleus number density in cm\ :sup:`-3`\ ."""
n_A          = n_H + n_He
"""Total hydrogen and helium nucleus number density in cm\ :sup:`-3`\ .""" 
chi          = n_He / n_H
"""Ratio of helium to hydrogen nucleus number."""
A_per_B      = n_A / n_B
"""Ratio of nucleus number to baryon number."""


#===== Cosmology =====

def hubble(rs, H0=H0, omega_m=omega_m, omega_rad=omega_rad, omega_lambda=omega_lambda):
    """Hubble parameter in s\ :sup:`-1`\ . (Copied from darkhistory.physics.hubble)

    Args:
        rs (float): The redshift of interest (rs = 1+z).
        H0 (float, optional): The Hubble parameter today, default value `H0`.
        omega_m (float, optional): Omega matter today, default value `omega_m`.
        omega_rad (float, optional): Omega radiation today, default value `omega_rad`.
        omega_lambda (float, optional): Omega dark energy today, default value `omega_lambda`.

    Returns:
        float: Hubble parameter in s\ :sup:`-1`\ .

    Notes:
        Assumes a flat universe.
    """

    return H0*np.sqrt(omega_rad*rs**4 + omega_m*rs**3 + omega_lambda)

def dtdz(rs, H0=H0, omega_m=omega_m, omega_rad=omega_rad, omega_lambda=omega_lambda):
    """dt/dz [s] (Based on darkhistory.physics.inj_rate)

    Args:
        rs (float): The redshift of interest (rs = 1+z).
        H0 (float, optional): The Hubble parameter today, default value `H0`.
        omega_m (float, optional): Omega matter today, default value `omega_m`.
        omega_rad (float, optional): Omega radiation today, default value `omega_rad`.
        omega_lambda (float, optional): Omega dark energy today, default value `omega_lambda`.

    Returns:
        float: dt/dz [s]
    """

    return -1./(rs*hubble(rs, H0, omega_m, omega_rad, omega_lambda))

# def dt_between_z(z_high, z_low):
#     """Calculate delta t [s] between z_high and z_low.
#     Alternative time step calculation, not in use.

#     Args:
#         z_high (float): Higher redshift.
#         z_low (float): Lower redshift.

#     Returns:
#         float: Delta t [s].
#     """
#     norm = dtdz(1+z_high)
#     integrand = lambda z: dtdz(1+z) / norm
#     val, err = np.array(integrate.quad(integrand, z_high, z_low)) * norm
#     return val

def conformal_dt_between_z(z_high, z_low):
    """Calculate conformal delta t [conformal s] between z_high and z_low.
    
    Args:
        z_high (float): Higher redshift.
        z_low (float): Lower redshift.

    Returns:
        float: Delta conformal t [conformal s].
    """
    norm = (1+z_high) * dtdz(1+z_high)
    integrand = lambda z: (1+z) * dtdz(1+z) / norm
    val, err = np.array(integrate.quad(integrand, z_high, z_low)) * norm
    return val

def conformal_dx_between_z(z_high, z_low):
    """Calculate conformal delta x [conformal Mpc] between z_high and z_low.
    Used for calculating shell radii.

    Args:
        z_high (float): Higher redshift.
        z_low (float): Lower redshift.

    Returns:
        float: Delta conformal x [conformal Mpc].
    """
    return np.abs(conformal_dt_between_z(z_high, z_low) * c / Mpc)

def dt_step(z, zplusone_factor):
    """Calculate delta t [s] between current redshift z and next redshift step.
    Consistent with 21cmFAST's time step.

    Args:
        z (float): Redshift.
        zplusone_factor (float): (1+z)/(1+z_next)

    Returns:
        float: Delta t [s].
    """
    z_next = (1+z) / zplusone_factor - 1
    return np.abs((cosmo.age(z) - cosmo.age(z_next)).to('s').value)


#===== Dark Matter =====
        
def struct_boost_func(model=...):
    """Structure formation boost factor 1+B(z). (Copied from darkhistory.physics.struct_boost_func)

    Args:
        model (str): Model to use. One of {'einasto_subs', 'einasto_no_subs',
            'NFW_subs', 'NFW_no_subs', 'erfc 1e-3', 'erfc 1e-6', 'erfc 1e-9'}.
    
    Returns:
        function: Boost factor function.

    Notes:
        Refer to 1408.1109 for erfc models, 1604.02457 for all other model descriptions and parameters.
    """

    if model.startswith('erfc'):
        from scipy.special import erfc

        if model == 'erfc 1e-3':
            b_h, z_h, delta = 1.6e5, 19.5, 1.54 # M_h,min = 1e-3 Msun
        elif model == 'erfc 1e-6':
            b_h, z_h, delta = 6.0e5, 19.0, 1.52 # M_h,min = 1e-6 Msun
        elif model == 'erfc 1e-9':
            b_h, z_h, delta = 2.3e6, 18.6, 1.48 # M_h,min = 1e-9 Msun
        else:
            raise ValueError(model)

        return lambda rs: 1 + b_h / rs**delta * erfc( rs/(1+z_h) )

    else:
        struct_data = dh_load_data('struct')[model]
        log_struct_interp = interpolate.interp1d(
            np.log(struct_data[:,0]), np.log(struct_data[:,1]),
            bounds_error=False, fill_value=(np.nan, 0.)
        )

        return lambda rs: np.exp(log_struct_interp(np.log(rs)))