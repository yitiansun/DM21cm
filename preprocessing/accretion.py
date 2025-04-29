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
from preprocessing import halo


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

def Mdot_PR(M, rho_inf, v, c_in, c_inf, lambda_fudge=1):
    """Park Ricotti accretion rate [eV/s]
    
    Args:
        M (float): Mass of the PBH [M_sun]
        rho_inf (float): Density of gas far away from the I-front [g/cm^3]
        v (float): Relative velocity of PBH to gas far away from the I-front [km/s]
        c_in (float): Sound speed inside the I-front [km/s]
        c_inf (float): Sound speed outside the I-front [km/s]
        lambda_fudge (float): Fudge factor in front of Mdot
    """
    rho_in, v_in = rho_in_v_in(rho_inf, v, c_in, c_inf)
    return _BHL_UNIT_FACTOR * lambda_fudge * M**2 * rho_in / (v_in**2 + c_in**2)**1.5

def Mdot_BHL(M, rho_inf, v, c_in, c_inf, lambda_fudge=1):
    """Bondi-Hoyle-Lyttleton accretion rate [eV/s]
    
    Args:
        M (float): Mass of the PBH [M_sun]
        rho_inf (float): Density of gas far away from the I-front [g/cm^3]
        v (float): Relative velocity of PBH to gas far away from the I-front [km/s]
        c_in (float): Sound speed inside the I-front [km/s]
        c_inf (float): Sound speed outside the I-front [km/s]
        lambda_fudge (float): Fudge factor in front of Mdot
    """
    return _BHL_UNIT_FACTOR * lambda_fudge * M**2 * rho_inf / (v**2 + c_inf**2)**1.5


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
    return 30 * jnp.minimum((1 + z) / 1000, 1)

def f_MB(v, v_rms):
    """Maxwell-Boltzmann distribution function, solid angle integrated [(km/s)^-1]
    
    Args:
        v (float): velocity [km/s]
        v_rms (float): root mean square velocity [km/s]
    """
    return (3 / (2 * jnp.pi * v_rms**2))**1.5 * 4 * jnp.pi * v**2 * jnp.exp(- 3 * v**2 / (2 * v_rms**2))



#===== PBH accretion model =====

RHO_DM = cosmo.Odm0 * cosmo.critical_density0.to(u.M_sun / u.Mpc**3).value # conformal dark matter density
F_DM = cosmo.Odm0 / cosmo.Om0
F_BM = cosmo.Ob0 / cosmo.Om0
KM_PER_PC = (1 * u.pc).to(u.km).value


class PBHAccretionModel:

    def __init__(self, accretion_type, c_in=None, lambda_fudge=1):
        """PBH accretion model
        
        Args:
            accretion_type (str): supports 'PR-ADAF', 'BHL-ADAF'
            c_in (float): Sound speed inside the I-front [km/s]. Required for PR models.
            lambda_fudge (float): Fudge factor in front of Mdot. Required for BHL models.
        """

        self.accretion_type = accretion_type
        self.c_in = c_in
        self.lambda_fudge = lambda_fudge

        if self.accretion_type == 'PR-ADAF':
            self.name = f'PRc{self.c_in:.0f}'
            self.Mdot_func = Mdot_PR
            self.L_func = L_ADAF
        elif self.accretion_type == 'BHL-ADAF':
            if self.lambda_fudge == 1:
                self.name = 'BHLl0'
            elif self.lambda_fudge == 1e-2:
                self.name = 'BHLl2'
            else:
                raise NotImplementedError(self.lambda_fudge)
            self.Mdot_func = Mdot_BHL # without fudge factor
            self.L_func = L_ADAF
        else:
            raise NotImplementedError(self.accretion_type)
        
        #===== vectorized and partially applied functions =====
        def Mdot_func_v(M_PBH, rho_inf, v, c_inf):
            return self.Mdot_func(M_PBH, rho_inf, v, self.c_in, c_inf, self.lambda_fudge)
        self.Mdot_func_v = jax.jit(jax.vmap(Mdot_func_v, in_axes=(None, None, 0, None)))

        def L_func_v(M_PBH, rho_inf, v, c_inf):
            Mdot = self.Mdot_func(M_PBH, rho_inf, v, self.c_in, c_inf, self.lambda_fudge) # [M_sun/yr]
            return self.L_func(Mdot, M_PBH)
        self.L_func_v = jax.jit(jax.vmap(L_func_v, in_axes=(None, None, 0, None)))

        #===== precomputed reference values =====
        self.rho_m_ref = 1 # reference physical total matter density in halos [M_sun/pc^3]
        self.m_PBH_ref = 1 # reference PBH mass [M_sun]
        self.n_PBH_ref = F_DM * (self.rho_m_ref * u.M_sun / u.pc**3 / (self.m_PBH_ref * u.M_sun)).to(u.pc**-3).value # [pc^-3]
        self.rho_inf_ref = F_BM * (self.rho_m_ref * u.M_sun / u.pc**3).to(u.g/u.cm**3).value # [g/cm^3]

        self.T_k_ref = 1 # reference gas kinetic temperature [eV]
        self.x_e_ref = 0 # reference ionization fraction
        self.c_inf_ref = np.sqrt(5/3 * (1 + self.x_e_ref) * self.T_k_ref * u.eV / c.m_p).to(u.km/u.s).value


    def L_cosmo_single_PBH(self, m_PBH, z, rho_inf, c_inf):
        """PBH accretion luminosity due to single unbound PBH [M_sun/yr]
        
        Args:
            m_PBH (float): Mass of the PBH [M_sun]
            z (float): Redshift
            rho_inf (float): Ambient gas density [g/cm^3]
            c_inf (float): Ambient gas sound speed [km/s]
        """
        v_cb = v_cb_cosmo(z) # [km/s]
        v_s = jnp.linspace(0.1 * v_cb, 10 * v_cb, 1000)
        fv_s = f_MB(v_s, v_cb)
        v_integrand = self.L_func_v(m_PBH, rho_inf, v_s, c_inf)
        return jnp.trapz(fv_s * v_integrand, v_s)
    
    def L_cosmo_single_PBH_std(self, m_PBH, z):
        """PBH accretion luminosity due to single unbound PBH in standard cosmology [M_sun/yr]
        
        Args:
            m_PBH (float): Mass of the PBH [M_sun]
            z (float): Redshift
        """
        rho_inf = (cosmo.critical_density(z) * cosmo.Ob(z)).to(u.g/u.cm**3).value # [g/cm^3]
        T_k = dh_phys.Tm_std(1+z) # [eV]
        x_e = dh_phys.xHII_std(1+z)
        c_inf = np.sqrt(5/3 * (1+x_e) * T_k * u.eV / c.m_p).to(u.km/u.s).value
        return self.L_cosmo_single_PBH(m_PBH, z, rho_inf, c_inf)
    
    def Mdot_cosmo_single_PBH(self, m_PBH, z, rho_inf, c_inf):
        """PBH accretion rate due to single unbound PBH [M_sun/yr]
        May be deprecated
        
        Args:
            m_PBH (float): Mass of the PBH [M_sun]
            z (float): Redshift
            rho_inf (float): Ambient gas density [g/cm^3]
            c_inf (float): Ambient gas sound speed [km/s]
        """
        v_cb = v_cb_cosmo(z) # [km/s]
        v_s = jnp.linspace(0.1 * v_cb, 10 * v_cb, 1000)
        fv_s = f_MB(v_s, v_cb)
        v_integrand = self.Mdot_func_v(m_PBH, rho_inf, v_s, c_inf)
        return jnp.trapz(fv_s * v_integrand, v_s)
    
    def Mdot_cosmo_single_PBH_std(self, m_PBH, z):
        """PBH accretion rate due to single unbound PBH in standard cosmology [M_sun/yr]
        May be deprecated
        
        Args:
            m_PBH (float): Mass of the PBH [M_sun]
            z (float): Redshift
        """
        rho_inf = (cosmo.critical_density(z) * cosmo.Ob(z)).to(u.g/u.cm**3).value # [g/cm^3]
        T_k = dh_phys.Tm_std(1+z) # [eV]
        x_e = dh_phys.xHII_std(1+z)
        c_inf = np.sqrt(5/3 * (1+x_e) * T_k * u.eV / c.m_p).to(u.km/u.s).value
        return self.Mdot_cosmo_single_PBH(m_PBH, z, rho_inf, c_inf)
        
    
    def L_cosmo_density(self, m_PBH, z, rho_dm, rho_b_inf, T_k):
        """PBH accretion luminosity conformal density [M_sun/yr/cMpc^3]
        
        Args:
            z (float): Redshift
            rho_dm (float): DM density (need to account for f_collapsed) [M_sun/cMpc^3]
            rho_b_inf (float): Ambient gas density [g/cm^3]
            T_k (float): Gas kinetic temperature [eV]
        """
        n_PBH = rho_dm / m_PBH # [1/cMpc^3]
        return n_PBH * self.L_cosmo_single_PBH(m_PBH, z, rho_b_inf, T_k)
    
    def L_cosmo_density_std(self, m_PBH, z, f_coll):
        """PBH accretion luminosity conformal density in standard cosmology [M_sun/yr/cMpc^3]
        
        Args:
            z (float): Redshift
            f_coll (float): Fraction of DM in collapsed halos
        """
        n_PBH = (1 - f_coll) * RHO_DM / m_PBH # [1/cMpc^3]
        return n_PBH * self.L_cosmo_single_PBH_std(m_PBH, z)
    
    
    def L_halo(self, m_PBH, m_halo, c_halo, z):
        """PBH accretion luminosity in a halo [M_sun/yr]
        
        Args:
            m_PBH (float): Mass of the PBH [M_sun]
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
            n_PBH = self.n_PBH_ref * (self.m_PBH_ref / m_PBH) * (rho / self.rho_m_ref) # [pc^-3]
            rho_inf = self.rho_inf_ref * (rho / self.rho_m_ref) # [g/cm^3]
            c_inf = jnp.sqrt(5/9) * v0 * KM_PER_PC # [km/s]
        
            # v integral
            v_pc_s = jnp.linspace(1e-3 * v0, 2*ve, 100)
            v_km_s = v_pc_s * KM_PER_PC
            f_s = halo.dm_dm_v_rel_dist_unnorm(v_km_s, ve*KM_PER_PC, v0*KM_PER_PC)
            f_s /= jnp.trapz(f_s, v_km_s)
            L_s = self.L_func_v(m_PBH, rho_inf, v_km_s, c_inf) # [M_sun/yr]
            L_s = jnp.nan_to_num(L_s)
            L_v_expn = jnp.trapz(L_s * f_s, v_km_s) # [M_sun/yr]

            return 4 * jnp.pi * r**2 * n_PBH * L_v_expn # [M_sun/yr / pc]
        
        return jnp.trapz(L_halo_integrand(r_arr), r_arr) # [M_sun/yr]
    

    #===== Cross check functions =====

    def Mdot_cosmo_single_PBH_singlevcb(self, m_PBH, z, rho_inf, c_inf):
        """PBH accretion rate due to single unbound PBH [M_sun/yr]
        DEBUG: Assuming a single value of v_cb. To reproduce Fig. 1 of AEGSSV24.
        
        Args:
            m_PBH (float): Mass of the PBH [M_sun]
            z (float): Redshift
            rho_inf (float): Ambient gas density [g/cm^3]
            c_inf (float): Ambient gas sound speed [km/s]
        """
        v_cb = v_cb_cosmo(z) # [km/s]
        return self.Mdot_func(m_PBH, rho_inf, v_cb, self.c_in, c_inf, self.lambda_fudge)
    
    def L_cosmo_single_PBH_singlevcb(self, m_PBH, z, rho_inf, c_inf):
        """PBH accretion luminosity due to single unbound PBH [M_sun/yr]
        
        Args:
            m_PBH (float): Mass of the PBH [M_sun]
            z (float): Redshift
            rho_inf (float): Ambient gas density [g/cm^3]
            c_inf (float): Ambient gas sound speed [km/s]
        """
        Mdot = self.Mdot_cosmo_single_PBH_singlevcb(m_PBH, z, rho_inf, c_inf) # [M_sun/yr]
        return self.L_func(Mdot, m_PBH)