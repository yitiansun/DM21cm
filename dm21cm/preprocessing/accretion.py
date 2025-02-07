import os
import sys

import numpy as np
from astropy import units as u
from astropy import constants as c
from astropy.cosmology import Planck18 as cosmo

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

sys.path.append(os.environ['DH_DIR'])
from darkhistory import physics as dh_phys

sys.path.append(os.environ['DM21CM_DIR'])
from dm21cm.preprocessing import halo


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

def e0_a(Mdot_MdotEdd, delta):
    """epsilon_0 and a according to Table 1 of Xie & Yuan 2012 (Radiative efficiency of hot accretion flows).
    
    Args:
        Mdot_MdotEdd (float): Mdot / Mdot_Edd = Mdot / (10*L_Edd/c^2)
        delta (float): electron heating fraction
    """
    ratio_range_data = {
        0.5:  jnp.array([2.9e-5, 3.3e-3, 5.3e-3]), # ratio values larger than the last will fail
        0.1:  jnp.array([9.4e-5, 5.0e-3, 6.6e-3]),
        1e-2: jnp.array([1.6e-5, 5.3e-3, 7.1e-3]),
        1e-3: jnp.array([7.6e-5, 4.5e-3, 7.1e-3]),
    }
    e0_a_data = {
        0.5:  jnp.array([[1.58 , 0.65], [0.055, 0.076], [0.17, 1.12], [0.0835, 0]]),
        0.1:  jnp.array([[0.12 , 0.59], [0.026, 0.27 ], [0.50, 4.53], [0.0761, 0]]),
        1e-2: jnp.array([[0.069, 0.69], [0.027, 0.54 ], [0.42, 4.85], [0.0798, 0]]),
        1e-3: jnp.array([[0.065, 0.71], [0.020, 0.47 ], [0.26, 3.67], [0.0740, 0]]),
    }
    i = jnp.searchsorted(ratio_range_data[delta], Mdot_MdotEdd)
    return e0_a_data[delta][i]


_L_EDD_UNIT_FACTOR = (4 * np.pi * c.G * u.M_sun * c.m_p / c.sigma_T / c.c).to(u.M_sun/u.yr).value

def L_Edd(M):
    """Eddington luminosity / c^2 [M_sun/yr]
    
    Args:
        M (float): Mass of the PBH [M_sun]
    """
    return _L_EDD_UNIT_FACTOR * M

def Mdot_Edd(M):
    """Eddington accretion rate [M_sun/yr]
    
    Args:
        M (float): Mass of the PBH [M_sun]
    """
    return 10 * L_Edd(M)

def L_ADAF(Mdot, M, delta=0.1):
    """ADAF luminosity / c^2 [M_sun/yr]
    
    Args:
        Mdot (float): Accretion rate [M_sun/yr]
        M (float): Mass of the PBH [M_sun]
        delta (float): electron heating fraction
    """
    Mdot_MdotEdd = Mdot / Mdot_Edd(M)
    e0, a = e0_a(Mdot_MdotEdd, delta)
    epsilon = e0 * (100 * Mdot_MdotEdd)**a
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



#===== PBH accretion model =====


# conformal dark matter density [M_sun / cMpc^3]
RHO_DM = cosmo.Odm0 * cosmo.critical_density0.to(u.M_sun / u.Mpc**3).value
F_DM = cosmo.Odm0 / cosmo.Om0
F_BM = cosmo.Ob0 / cosmo.Om0
KM_PER_PC = (1 * u.pc).to(u.km).value

#--- jaxified functions ---
# Mdot_BHL_v = jax.jit(jax.vmap(Mdot_BHL, in_axes=(None, None, 0, None, None)))
# Mdot_PR_v = jax.jit(jax.vmap(Mdot_PR, in_axes=(None, None, 0, None, None)))

# @jax.jit
# @partial(jax.vmap, in_axes=(None, None, 0, None, None))
# def L_ADAF_BHL_v(M_PBH, rho_inf, v, c_in, c_inf):
#     Mdot = Mdot_BHL(M_PBH, rho_inf, v, c_in, c_inf) # [M_sun/yr]
#     return L_ADAF(Mdot, M_PBH)

# @jax.jit
# @partial(jax.vmap, in_axes=(None, None, 0, None, None))
# def L_ADAF_PR_v(M_PBH, rho_inf, v, c_in, c_inf):
#     Mdot = Mdot_PR(M_PBH, rho_inf, v, c_in, c_inf) # [M_sun/yr]
#     return L_ADAF(Mdot, M_PBH)


class PBHAccretionModel:

    def __init__(self, m_PBH, f_PBH, accretion_type, c_in=None):
        """PBH accretion model
        
        Args:
            m_PBH (float): PBH mass [M_sun]
            f_PBH (float): PBH fraction of DM
            accretion_type (str): supports 'PR-ADAF', 'BHL-ADAF'
            c_in (float): Sound speed inside the I-front [km/s]. Required for PR models.
        """

        self.m_PBH = m_PBH
        self.f_PBH = f_PBH
        self.accretion_type = accretion_type
        self.c_in = c_in

        if self.accretion_type == 'PR-ADAF':
            self.Mdot_func = Mdot_PR
            self.L_func = L_ADAF
        elif self.accretion_type == 'BHL-ADAF':
            self.Mdot_func = Mdot_BHL
            self.L_func = L_ADAF
        else:
            raise NotImplementedError(self.accretion_type)
        
        self.Mdot_func_v = jax.jit(jax.vmap(self.Mdot_func, in_axes=(None, None, 0, None, None)))
        def L_func_full(M_PBH, rho_inf, v, c_in, c_inf):
            Mdot = self.Mdot_func(M_PBH, rho_inf, v, c_in, c_inf) # [M_sun/yr]
            return self.L_func(Mdot, M_PBH)
        self.L_func_v = jax.jit(jax.vmap(L_func_full, in_axes=(None, None, 0, None, None)))

        self.rho_m_ref = 1 # reference physical total matter density in halos [M_sun/pc^3]
        self.n_PBH_ref = self.f_PBH * F_DM * (self.rho_m_ref * u.M_sun / u.pc**3 / (self.m_PBH * u.M_sun)).to(u.pc**-3).value # [pc^-3]
        self.rho_inf_ref = F_BM * (self.rho_m_ref * u.M_sun / u.pc**3).to(u.g/u.cm**3).value # [g/cm^3]



    def L_cosmo_single_PBH(self, z):
        """PBH accretion luminosity due to single PBH not in halos [M_sun/yr]
        
        Args:
            z (float): Redshift
        """
        rho_inf = (cosmo.critical_density(z) * cosmo.Ob(z)).to(u.g/u.cm**3).value # [g/cm^3]
        T_K = dh_phys.Tm_std(1+z) # [eV]
        c_inf = np.sqrt(5/3 * T_K * u.eV / c.m_p).to(u.km/u.s).value # [km/s]
        v_cb = v_cb_cosmo(z) # [km/s]
        v_s = jnp.linspace(0.1 * v_cb, 10 * v_cb, 1000)
        fv_s = f_MB(v_s, v_cb)
        v_integrand = self.L_func_v(self.m_PBH, rho_inf, v_s, self.c_in, c_inf)
        return jnp.trapz(fv_s * v_integrand, v_s)
    
    def Mdot_cosmo_single_PBH(self, z):
        """PBH accretion rate due to single PBH not in halos [M_sun/yr]
        May be deprecated
        
        Args:
            z (float): Redshift
        """
        rho_inf = (cosmo.critical_density(z) * cosmo.Ob(z)).to(u.g/u.cm**3).value # [g/cm^3]
        T_K = dh_phys.Tm_std(1+z) # [eV]
        c_inf = np.sqrt(5/3 * T_K * u.eV / c.m_p).to(u.km/u.s).value # [km/s]
        v_cb = v_cb_cosmo(z) # [km/s]
        v_s = jnp.linspace(0.1 * v_cb, 10 * v_cb, 1000)
        fv_s = f_MB(v_s, v_cb)
        v_integrand = self.Mdot_func_v(self.m_PBH, rho_inf, v_s, self.c_in, c_inf)
        return jnp.trapz(fv_s * v_integrand, v_s)
    
    def L_cosmo_density(self, z, f_coll):
        """PBH accretion luminosity conformal density [M_sun/yr/cMpc^3]
        
        Args:
            z (float): Redshift
            f_coll (float): Collapsed fraction
        """
        n_PBH = (1 - f_coll) * self.f_PBH * RHO_DM / self.m_PBH # [1/cMpc^3]
        return n_PBH * self.L_cosmo_single_PBH(z)
    
    
    def L_halo(self, m_halo, c_halo, z):
        """PBH accretion luminosity in a halo [M_sun/yr]
        
        Args:
            m_halo (float): Mass of the halo [M_sun]
            c_halo (float): Halo concentration parameter
            z (float): Redshift
        """

        rho_s, r_s, r_delta = halo.nfw_info(m_halo, c_halo, z)
        r_arr = jnp.geomspace(1e-4 * r_s, 0.99 * r_delta, 300)

        @partial(jax.vmap, in_axes=(0,))
        def L_halo_integrand(r):
            
            # v-dist parameters
            ve = halo.get_ve(r, rho_s, r_s, r_delta) # [pc/s]
            v0 = halo.get_v0_jeans(r, rho_s, r_s, r_delta) # [pc/s]
            rho = halo.nfw_density(r, rho_s, r_s) # [M_sun/pc^3]

            # accretion parameters
            n_PBH = self.n_PBH_ref * (rho / self.rho_m_ref) # [pc^-3]
            rho_inf = self.rho_inf_ref * (rho / self.rho_m_ref) # [g/cm^3]
            c_inf = jnp.sqrt(5/9) * v0 * KM_PER_PC # [km/s]
        
            # v integral
            v_pc_s = jnp.linspace(1e-3 * v0, ve, 100)
            v_km_s = v_pc_s * KM_PER_PC
            f_s = halo.dm_rest_v_rel_dist_unnorm(v_pc_s, ve, v0)
            f_s /= jnp.trapz(f_s, v_km_s)
            L_s = self.L_func_v(self.m_PBH, rho_inf, v_km_s, self.c_in, c_inf) # [M_sun/yr]
            L_s = jnp.nan_to_num(L_s)
            L_v_expn = jnp.trapz(L_s * f_s, v_km_s) # [M_sun/yr]

            return 4 * jnp.pi * r**2 * n_PBH * L_v_expn # [M_sun/yr / pc]
        
        return jnp.trapz(L_halo_integrand(r_arr), r_arr) # [M_sun/yr]