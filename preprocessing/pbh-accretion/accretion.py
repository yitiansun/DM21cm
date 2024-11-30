import numpy as np
from astropy import units as u
from astropy import constants as c

import jax
import jax.numpy as jnp
from functools import partial


#===== M dot: PR and BHL =====

def v_R_v_D(c_in, c_inf):
    """Returns v_R, v_D [km/s]
    
    Args:
        c_in (float): Sound speed inside the I-front [km/s]
        c_inf (float): Sound speed outside the I-front [km/s]
    """
    discriminant = jnp.sqrt(c_in**2 - c_inf**2)
    return jnp.array([
        jnp.sqrt(2 * c_in**2 - c_inf**2 + 2 * c_in * discriminant),
        jnp.sqrt(2 * c_in**2 - c_inf**2 - 2 * c_in * discriminant)
    ])

def sqrtDelta(v, c_in, c_inf):
    """Returns sqrt(Delta) [km/s] (A.5)
    
    Args:
        v (float): Relative velocity of PBH to gas far away from the I-front [km/s]
        c_in (float): Sound speed inside the I-front [km/s]
        c_inf (float): Sound speed outside the I-front [km/s]
    """
    return jnp.sqrt((v**2 + c_inf**2)**2 - 4 * v**2 * c_in**2)

def rho_in_v_in_high(rho_inf, v, c_in, c_inf):
    ratio = (v**2 + c_inf**2 - sqrtDelta(v, c_in, c_inf)) / (2 * c_in**2)
    rho_in = rho_inf * ratio
    v_in = v / ratio
    return jnp.array([rho_in, v_in])

def rho_in_v_in_low(rho_inf, v, c_in, c_inf):
    ratio = (v**2 + c_inf**2 + sqrtDelta(v, c_in, c_inf)) / (2 * c_in**2)
    rho_in = rho_inf * ratio
    v_in = v / ratio
    return jnp.array([rho_in, v_in])

def rho_in_v_in_mid(rho_inf, v, c_in, c_inf):
    ratio = (v**2 + c_inf**2) / (2 * c_in**2)
    rho_in = rho_inf * ratio
    v_in = c_in
    return jnp.array([rho_in, v_in])

def rho_in_v_in(rho_inf, v, c_in, c_inf):
    """Returns rho_in [g/cm^3], v_in [km/s] (A.2-4)
    
    Args:
        rho_inf (float): Density of gas far away from the I-front [g/cm^3]
        v (float): Relative velocity of PBH to gas far away from the I-front [km/s]
        c_in (float): Sound speed inside the I-front [km/s]
        c_inf (float): Sound speed outside the I-front [km/s]
    """
    v_R, v_D = v_R_v_D(c_in, c_inf)
    return jnp.where(v < v_D,
        rho_in_v_in_low(rho_inf, v, c_in, c_inf),
        jnp.where(v < v_R,
            rho_in_v_in_mid(rho_inf, v, c_in, c_inf),
            rho_in_v_in_high(rho_inf, v, c_in, c_inf)
        )
    )


_BHL_UNIT_FACTOR = 4 * np.pi * (c.G**2 * c.M_sun**2 * u.g/u.cm**3 / (u.km/u.s)**3).to(u.M_sun/u.yr).value
# Mdot [# in eV/s] = BHL_unit_factor * M [M_sun]**2 * rho [g/cm^3] / v [km/s]**3

def Mdot_PR(M, rho_inf, v, c_in, c_inf):
    """Park Ricotti accretion rate [eV/s]
    
    Args:
        M (float): Mass of the PBH [M_sun]
        rho_inf (float): Density of gas far away from the I-front [g/cm^3]
        v (float): Relative velocity of PBH to gas far away from the I-front [km/s]
        c_in (float): Sound speed inside the I-front [km/s]
        c_inf (float): Sound speed outside the I-front [km/s]
    """
    rho_in, v_in = rho_in_v_in(rho_inf, v, c_in, c_inf)
    return _BHL_UNIT_FACTOR * M**2 * rho_in / (v_in**2 + c_in**2)**1.5

def Mdot_BHL(M, rho_inf, v, c_in, c_inf):
    """Bondi-Hoyle-Lyttleton accretion rate [eV/s]
    
    Args:
        M (float): Mass of the PBH [M_sun]
        rho_inf (float): Density of gas far away from the I-front [g/cm^3]
        v (float): Relative velocity of PBH to gas far away from the I-front [km/s]
        c_in (float): Sound speed inside the I-front [km/s]
        c_inf (float): Sound speed outside the I-front [km/s]
    """
    return _BHL_UNIT_FACTOR * M**2 * rho_inf / (v**2 + c_inf**2)**1.5


#===== Luminosity: ADAF & Thin disk =====

def e0_a(ratio, delta):
    """epsilon_0 and a according to Table 1 of Xie & Yuan 2012 (Radiative efficiency of hot accretion flows).
    
    Args:
        ratio (float): ratio = Mdot / Mdot_Edd = Mdot / (10*L_Edd/c^2)
        delta (float): electron heating fraction
    """
    ratio_range_data = {
        0.5:  jnp.array([2.9e-5, 3.3e-3, 5.3e-3]), # ratio values larger than the last will fail
        0.1:  jnp.array([9.4e-5, 5.0e-3, 6.6e-3]),
        1e-2: jnp.array([1.6e-5, 5.3e-3, 7.1e-3]),
        1e-3: jnp.array([7.6e-5, 4.5e-3, 7.1e-3]),
    }
    e0_a_data = {
        0.5:  jnp.array([[1.58 , 0.65], [0.055, 0.076], [0.17, 1.12]]),
        0.1:  jnp.array([[0.12 , 0.59], [0.026, 0.27 ], [0.50, 4.53]]),
        1e-2: jnp.array([[0.069, 0.69], [0.027, 0.54 ], [0.42, 4.85]]),
        1e-3: jnp.array([[0.065, 0.71], [0.020, 0.47 ], [0.26, 3.67]]),
    }
    i = jnp.searchsorted(ratio_range_data[delta], ratio)
    # if i == 3:
    #     raise ValueError('ratio is too large')
    return e0_a_data[delta][i]


_L_EDD_UNIT_FACTOR = (4 * np.pi * c.G * u.M_sun * c.m_p / c.sigma_T / c.c).to(u.M_sun/u.yr).value

def L_Edd(M):
    """Eddington luminosity / c^2 [M_sun/yr]
    
    Args:
        M (float): Mass of the PBH [M_sun]
    """
    return _L_EDD_UNIT_FACTOR * M

def L_ADAF(Mdot, M, delta=0.1):
    """ADAF luminosity / c^2 [M_sun/yr]
    
    Args:
        Mdot (float): Accretion rate [M_sun/yr]
        M (float): Mass of the PBH [M_sun]
        delta (float): electron heating fraction
    """
    ratio = Mdot / (10 * L_Edd(M))
    e0, a = e0_a(ratio, delta)
    epsilon = e0 * ratio**a
    return epsilon * Mdot

def L_thin(Mdot, M):
    """Thin disk accretion luminosity / c^2 [M_sun/yr]
    
    Args:
        Mdot (float): Accretion rate [M_sun/yr]
        M (float): Mass of the PBH [M_sun]
    """
    return 0.1 * Mdot


#===== Velocity dispersion =====

def v_cb_cosmo(z):
    """DM streaming velocity [km/s]
    
    Args:
        z (float): Redshift
    """
    return 30 * (1 + z) / 1000

def f_MB(v, v_rms):
    """Maxwell-Boltzmann distribution function, solid angle integrated [(km/s)^-1]
    
    Args:
        v (float): velocity [km/s]
        v_rms (float): root mean square velocity [km/s]
    """
    return (3 / (2 * jnp.pi * v_rms**2))**1.5 * 4 * jnp.pi * v**2 * jnp.exp(- 3 * v**2 / (2 * v_rms**2))