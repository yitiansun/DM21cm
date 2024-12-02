import numpy as np
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import Planck18 as cosmo

import jax
import jax.numpy as jnp
import jax.scipy as jsp

#===== Constants =====

_G = c.G.to(u.pc**3/ u.M_sun / u.s**2).value # [pc^3 / M_sun / s^2]
_C = c.c.to(u.pc/u.s).value # [pc / s]

_RHO_C_Z_ARR = jnp.linspace(0, 100, 1000)
_RHO_C_ARR = cosmo.critical_density(_RHO_C_Z_ARR).to('M_sun / pc^3').value # [M_sun / pc^3]

def get_rho_c(z):
    """Jit-able critical density [M_sun / pc^3]"""
    return jnp.interp(z, _RHO_C_Z_ARR, _RHO_C_ARR)


#===== NFW =====

def nfw_info(M, c, z):
    """Returns rho_s [M_sun / pc^3], r_s [pc], r_delta [pc]
    
    Args:
        M (float): Mass of the halo [M_sun]
        c (float): Concentration parameter
        z (float): Redshift
    """
    rho_c = get_rho_c(z) # [M_sun / pc^3]
    delta = 200
    r_s = ((3 * M)/(4 * jnp.pi * c**3 * delta * rho_c))**(1/3) # [pc]
    r_delta = c * r_s # [pc]
    rho_s = ((1 + c)*M)/(16 * jnp.pi * r_s**3 * (-c + jnp.log(1 + c) + c*jnp.log(1 + c))) # [M_sun / pc^3]
    return rho_s, r_s, r_delta

def nfw_density(r, rho_s, r_s):
    """Returns rho [M_sun / pc^3]
    
    Args:
        r (float): Radius [pc]
        rho_s (float): Density at scale radius [M_sun / pc^3]
        r_s (float): Scale radius [pc]
    """
    return 4 * rho_s / ((r/r_s) * (1 + r/r_s)**2)

def nfw_enclosed(r, rho_s, r_s):
    """Returns M_enc [M_sun]. Args see nfw_density."""
    return 16 * jnp.pi * r_s**3 * rho_s * (-(r / (r + r_s)) - jnp.log(r_s) + jnp.log(r + r_s))

def nfw_phi(r, rho_s, r_s):
    """Returns phi [pc^2 / s^2]. Args see nfw_density."""
    return - 16 * jnp.pi * _G * rho_s * r_s**3 / r * jnp.log(1 + r / r_s)


#===== Jeans analysis with MB approximation =====
# Following 2005.03955

def jeans_integrand(r, rho_s, r_s):
    """Returns rho * M_enc / r^2 [M_sun^2 / pc^5]. Args see nfw_density."""
    return nfw_density(r, rho_s, r_s) * nfw_enclosed(r, rho_s, r_s) / r**2

def get_sigma_v(r, rho_s, r_s, r_delta):
    """Returns sigma_v [pc / s].
    
    Args:
        r_delta (float): Radius at Delta [pc]
        See nfw_density for other args.
    """
    logr_arr = jnp.linspace(jnp.log(r), jnp.log(r_delta), 300)
    r_arr = jnp.exp(logr_arr)
    integral = jnp.trapz(r_arr * jeans_integrand(r_arr, rho_s, r_s), logr_arr) # [M_sun^2 / pc^4]
    v_disp = 3 * _G / nfw_density(r, rho_s, r_s) * integral # [pc^2 / s^2]
    return jnp.sqrt(v_disp)

def get_ve(r, rho_s, r_s, r_delta):
    """Returns v_e [pc / s]. Args see get_sigma_v."""
    psi = nfw_phi(r_delta, rho_s, r_s)- nfw_phi(r, rho_s, r_s) # [pc^2 / s^2]
    return jnp.sqrt(2*psi)

def rel_v_disp(r, rho_s, r_s, r_delta):
    """Returns <v_rel^2> [pc^2 / s^2]. Args see get_sigma_v."""
    ve = get_ve(r, rho_s, r_s, r_delta) # [pc / s]
    sigma_v = get_sigma_v(r, rho_s, r_s, r_delta) # [pc / s]
    v0 = jnp.sqrt(2/3) * sigma_v # [pc / s]
    vsq = 3 * v0**2 + (24/5 * ve**5) / (6 * v0**2 * ve + 4 * ve**3 - 3 * jnp.exp((ve/v0)**2) * jnp.sqrt(jnp.pi) * v0**3 * jsp.special.erf(ve/v0)) # [pc^2 / s^2]
    return vsq