"""Functions for halo properties"""

import os
import sys
import numpy as np
from scipy import interpolate
import astropy.units as u
import astropy.constants as c
from astropy.cosmology import Planck18 as cosmo

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy as jsp

import halomod

sys.path.append(os.environ['DM21CM_DIR'])
from dm21cm.utils import load_h5_dict
from dm21cm.interpolators import interp1d, bound_action

#===== Constants =====

_G = c.G.to(u.pc**3/ u.M_sun / u.s**2).value # [pc^3 / M_sun / s^2]
# _C = c.c.to(u.pc/u.s).value # [pc / s]
DM_FRAC = cosmo.Odm0 / (cosmo.Odm0+cosmo.Ob0)

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
    """NFW density [M_sun / pc^3]
    
    Args:
        r (float): Radius [pc]
        rho_s (float): Density at scale radius [M_sun / pc^3]
        r_s (float): Scale radius [pc]
    """
    return 4 * rho_s / ((r/r_s) * (1 + r/r_s)**2)

def nfw_enclosed(r, rho_s, r_s):
    """M_enc [M_sun]. Args see nfw_density."""
    return 16 * jnp.pi * r_s**3 * rho_s * (-(r / (r + r_s)) - jnp.log(r_s) + jnp.log(r + r_s))

def nfw_phi(r, rho_s, r_s):
    """phi [pc^2 / s^2]. Args see nfw_density."""
    return - 16 * jnp.pi * _G * rho_s * r_s**3 / r * jnp.log(1 + r / r_s)


#===== Jeans analysis with MB approximation =====
# Following 2005.03955

def jeans_integrand(r, rho_s, r_s):
    """Returns rho * M_enc / r^2 [M_sun^2 / pc^5]. Args see nfw_density."""
    return nfw_density(r, rho_s, r_s) * nfw_enclosed(r, rho_s, r_s) / r**2

def get_sigma_v(r, rho_s, r_s, r_delta):
    """Returns sigma_v [pc / s].
    
    Args:
        r (float): Radius [pc]
        rho_s (float): Density at scale radius [M_sun / pc^3]
        r_s (float): Scale radius [pc]
        r_delta (float): Radius at Delta [pc]
    """
    logr_arr = jnp.linspace(jnp.log(r), jnp.log(r_delta), 1000)
    r_arr = jnp.exp(logr_arr)
    integral = jnp.trapz(r_arr * jeans_integrand(r_arr, rho_s, r_s), logr_arr) # [M_sun^2 / pc^4]
    v_disp = 3 * _G / nfw_density(r, rho_s, r_s) * integral # [pc^2 / s^2]
    return jnp.sqrt(v_disp)

def get_ve(r, rho_s, r_s, r_delta):
    """Returns v_e [pc / s]. Args see get_sigma_v."""
    psi = nfw_phi(r_delta, rho_s, r_s) - nfw_phi(r, rho_s, r_s) # [pc^2 / s^2]
    return jnp.sqrt(2*psi)

def get_v0_jeans(r, rho_s, r_s, r_delta):
    """Returns v0 [pc / s] for Jeans model (2.21). Args see get_sigma_v."""
    return jnp.sqrt(2/3) * get_sigma_v(r, rho_s, r_s, r_delta)

def rel_v_disp(r, rho_s, r_s, r_delta):
    """<v_rel^2> [pc^2 / s^2]. Args see get_sigma_v."""
    ve = get_ve(r, rho_s, r_s, r_delta) # [pc / s]
    sigma_v = get_sigma_v(r, rho_s, r_s, r_delta) # [pc / s]
    v0 = jnp.sqrt(2/3) * sigma_v # [pc / s]
    vsq = 3 * v0**2 + (24/5 * ve**5) / (6 * v0**2 * ve + 4 * ve**3 - 3 * jnp.exp((ve/v0)**2) * jnp.sqrt(jnp.pi) * v0**3 * jsp.special.erf(ve/v0)) # [pc^2 / s^2]
    return vsq

def dm_rest_v_rel_dist_unnorm(v, ve, v0):
    """Unnormalized DM-rest |v_rel| distribution v^2 f(v) [(pc / s)^-1] (2.20).
    
    Args:
        v (float): Velocity [pc / s]
        ve (float): Escape velocity [pc / s]
        v0 (float): Characteristic velocity [pc / s]
    """
    f = jnp.exp(- v**2 / v0**2) - jnp.exp(- ve**2 / v0**2)
    return v**2 * jnp.clip(f, 0, None)

def dm_dm_v_rel_dist_unnorm(v, ve, v0):
    """Unnormalized DM-DM |v_rel| distribution v^2 f(v) [(pc / s)^-1] (App.A).
    
    Args:
        v (float): Relative velocity [pc / s]
        ve (float): Escape velocity [pc / s]
        v0 (float): Characteristic velocity [pc / s]
    """
    b = ve / v0
    u = v / v0
    f = jnp.sqrt(8*jnp.pi) * jnp.exp(-u**2/2) * jsp.special.erf((2*b-u)/jnp.sqrt(2)) \
    + 4 / u * jnp.exp(-2*b**2) * (jnp.exp((2*b-u)*u) - 1) \
    + 8 * jnp.sqrt(jnp.pi) * jnp.exp(-b**2) * (jsp.special.erf(u-b) - jsp.special.erf(b)) \
    + 2 / 3 * jnp.exp(-2*b**2) * (16*b**3 - 12*b**2*u + u**3 + 24*b - 12*u)
    return v**2 * jnp.nan_to_num(jnp.clip(f, 0, None))


#===== HMF utils =====

def fix_cmz_numerical_issues(xs, ys, z):

    def accepted_lowz(xs, ys):
        m_max = 1.05
        m_min = 0.95
        xs_accepted = []
        ys_accepted = []

        for i, (x, y) in enumerate(zip(xs, ys)):
            if i == 0:
                xs_accepted.append(x)
                ys_accepted.append(y)
                continue
            if m_min < y/ys_accepted[-1] and y/ys_accepted[-1] < m_max:
                xs_accepted.append(x)
                ys_accepted.append(y)
        return xs_accepted, ys_accepted
    
    def accepted_highz(xs, ys, z):
        z_s = np.array([20, 30, 40, 50, 60, 70, 80, 90, 100])
        p0_s = np.array([3, 2.52, 2.35, 2.3, 2.2, 2.17, 2.14, 2.1, 2.1])
        p1_s = np.array([-0.055, -0.03, -0.025, -0.02, -0.015, -0.013, -0.012, -0.01, -0.01])
        p0 = np.interp(z, z_s, p0_s)
        p1 = np.interp(z, z_s, p1_s)
        y_upper = p0 + np.log10(xs) * p1
        xs_accepted = xs[ys < y_upper]
        ys_accepted = ys[ys < y_upper]
        return xs_accepted, ys_accepted

    xas, yas = accepted_lowz(xs, ys) if z < 20 else accepted_highz(xs, ys, z)
    log_ys_fixed = interpolate.interp1d(np.log(xas), np.log(yas), kind='linear', bounds_error=False, fill_value=np.min(np.log(ys)))(np.log(xs))
    return xs, np.exp(log_ys_fixed)

def cmz_raw(m, z, model='Ludlow16'):
    """CMZ relation interpolator.
    
    Args:
        m (array): Array of halo mass to interpolate over [Msun].
        z (float): Redshift.
        model (str, optional): Concentration model. Defaults to 'Ludlow16'.
    """
    hm = halomod.DMHaloModel(
        halo_concentration_model=model,
        z = z, Mmin = 0., Mmax = 19, dlog10m = 0.025,
        mdef_model='SOCritical', halo_profile_model = halomod.profiles.NFW
    )
    hm_m_s, hm_cmz_s = fix_cmz_numerical_issues(hm.m, hm.cmz_relation)
    return np.interp(np.log(m), np.log(hm_m_s), hm_cmz_s)


_CMZ_DATA = load_h5_dict(f"{os.environ['DM21CM_DIR']}/data/production/cmz_Ludlow16.h5")
_CMZ_M = _CMZ_DATA['m']
_CMZ_Z = _CMZ_DATA['z']
_CMZ_C = _CMZ_DATA['c']

def cmz(m, z):
    """CMZ relation interpolator from table.
    
    Args:
        m (array): Array of halo mass to interpolate over [Msun].
        z (float): Redshift.
    """
    z_in = bound_action(z, _CMZ_Z, 'clip')
    m_in = bound_action(m, _CMZ_M, 'clip')
    return interp1d(interp1d(_CMZ_C, _CMZ_Z, z_in), _CMZ_M, m_in)