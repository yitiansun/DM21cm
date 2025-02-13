"""Halo mass function evaluation."""

import os
import sys
from tqdm import tqdm

from astropy import units as u
from astropy import constants as c
from astropy.cosmology import Planck18 as cosmo
import numpy as np
from scipy import integrate, interpolate

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

from hmf import Transfer

sys.path.append(os.environ['DM21CM_DIR'])
from dm21cm.utils import load_h5_dict, save_h5_dict


#===== Constants =====
# conformal matter density [M_sun / cMpc^3]
RHO_M = cosmo.Om0 * cosmo.critical_density0.to(u.M_sun / u.Mpc**3).value

# Default Duty Cycle Parametrization
M_TURN = 5.012e8
M_MIN = 1 # [M_sun]
M_MAX = 2.211912255032272e+19 # [M_sun] | mass in a sphere of radius 512 cMpc


#===== Window functions =====

class SphereWindow:
    """From 21cmFAST."""

    def MtoR(self, M):
        """R: [cMpc] <- M: [M_sun]."""
        return (3 * M / (4 * jnp.pi * RHO_M)) ** (1/3)

    def RtoM(sefl, R):
        """M: [M_sun] <- R: [cMpc]."""
        return 4 * jnp.pi * RHO_M * R**3 / 3

    def W(self, k, R):
        """Top hat window function."""
        kR = k * R
        return jnp.where(kR >= 1e-4, 3.0 * (jnp.sin(kR) / kR**3 - jnp.cos(kR) / kR**2), 0.)

    def dWdR(self, k, R):
        """Derivative of top hat window function."""
        kR = k * R
        return jnp.where(k >= 1e-10, (9 * jnp.cos(kR) * k / kR**3) + (3 * jnp.sin(kR) * (1 - 3 / (kR * kR)) / (kR * R)), 0.)
    
class CubeWindow:
    """Side legnth = 2R."""

    def MtoR(self, M):
        """R: [cMpc] <- M: [M_sun]."""
        return (M / RHO_M) ** (1/3) / 2

    def RtoM(self, R):
        """M: [M_sun] <- R: [cMpc]."""
        return 8 * RHO_M * R**3

    def Wi(self, ki, R):
        kiR = ki * R
        return jnp.where(kiR >= 1e-10, jnp.sin(kiR) / (kiR), 0.0)

    def W(self, kx, ky, kz, R):
        return self.Wi(kx, R) * self.Wi(ky, R) * self.Wi(kz, R)

    def dWidR(self, ki, R):
        kiR = ki * R
        return jnp.where(ki >= 1e-10, (kiR*jnp.cos(kiR) - jnp.sin(kiR)) / (kiR*R), 0.0)

    def dWdR(self, kx, ky, kz, R):
        return self.dWidR(kx, R) * self.Wi(ky, R) * self.Wi(kz, R) + \
            self.Wi(kx, R) * self.dWidR(ky, R) * self.Wi(kz, R) + \
            self.Wi(kx, R) * self.Wi(ky, R) * self.dWidR(kz, R)
    

#===== Integrators =====

class CubeIntegrator:

    def __init__(self, k, res=None, use_tqdm=False):
        """Abstract cube integration class.
        Implement the integrand method in a subclass and call integral.
        
        Args:
            k (array): Array of k values.
            res (int, optional): Resolution of k values. If None, k is used as is.
            use_tqdm (bool, optional): Whether to use tqdm for integration.
        """
        self.res = res
        if self.res:
            self.k = jnp.geomspace(k[0], k[-1], self.res)
        else:
            self.k = jnp.asarray(k)
        self.lnk = jnp.log(self.k)
        self.use_tqdm = use_tqdm

    def integrand(self, kx, ky, kz, R):
        pass
    
    @partial(jax.vmap, in_axes=(None, None, 0, None))
    def integrand_2(self, kx, ky, R):
        return jnp.trapz(self.k * self.integrand(kx, ky, self.k, R), self.lnk)
    
    @partial(jax.jit, static_argnums=(0,))
    def integrand_1(self, kx, R):
        return jnp.trapz(self.k * self.integrand_2(kx, self.k, R), self.lnk)
    
    def integral(self, R):
        integrand = []
        k_iter = tqdm(self.k) if self.use_tqdm else self.k
        for k in k_iter:
            integrand.append(self.integrand_1(k, R))
        integrand = jnp.asarray(integrand)
        return jnp.trapz(self.k * integrand, self.lnk) * 8 # eight octants
    

#===== Power spectrum =====

class NormalizedPowerSpectrum:

    def __init__(self):
        """Power spectrum normalized to cosmo's sigma8."""

        self.window = SphereWindow()

        tr = Transfer(cosmo_model=cosmo, transfer_model='EH', z=0)
        k_s = tr.k * cosmo.h # to u without `h`
        T_s = tr.transfer_function / np.amax(tr.transfer_function) # normalized
        n = cosmo.to_format('mapping')['meta']['n']
        k2P_s = k_s**2 * T_s**2 * k_s**n
        k2P_unnorm = interpolate.CubicSpline(k_s, k2P_s)

        radius_8 = 8.0/cosmo.h
        sigma8sq_integrand = lambda logk: np.exp(logk)* k2P_unnorm(np.exp(logk)) * self.window.W(np.exp(logk), radius_8)**2
        sigma8sq = integrate.quad(sigma8sq_integrand, np.log(k_s[0]), np.log(k_s[-1]), epsabs=0, epsrel=1e-6)[0]

        sigma_norm = cosmo.to_format('mapping')['meta']['sigma8']/np.sqrt(sigma8sq)
        self.k_s = jnp.asarray(k_s)
        self.k2P_s = jnp.asarray(sigma_norm**2 * k2P_s)

    def k2P(self, k):
        """This k^2 P(k) carries some normalization such that int dk k2P(k) W(k, R)^2 = sigma^2."""
        return jnp.interp(k, self.k_s, self.k2P_s)
    

#===== Interpolators for sigma and dsigmasqdm =====

class SigmaMInterpSphere:

    def __init__(self, res=4001):

        self.name = 'sphere'

        ps = NormalizedPowerSpectrum()
        w = SphereWindow()
        k_s = jnp.geomspace(ps.k_s[0], ps.k_s[-1], 1000)
        k2P_s = ps.k2P(k_s)

        @jax.jit
        @jax.vmap
        def sigma_z0_raw(M):
            R = w.MtoR(M)
            sigmasq = jnp.trapz(k_s * k2P_s * w.W(k_s, R)**2, np.log(k_s))
            return jnp.sqrt(sigmasq)

        @jax.jit
        @jax.vmap
        def dsigmasqdm_z0_raw(M):
            R = w.MtoR(M)
            dRdM = R / (3*M)
            return jnp.trapz(k_s * k2P_s * 2 * w.W(k_s, R) * w.dWdR(k_s, R) * dRdM, np.log(k_s))

        self.m_s = jnp.geomspace(M_MIN, M_MAX, res)
        self.sigma_s = sigma_z0_raw(self.m_s)
        self.dsigmasqdm_s = dsigmasqdm_z0_raw(self.m_s)

    def sigma_z0(self, M):
        return jnp.interp(M, self.m_s, self.sigma_s)
    
    def dsigmasqdm_z0(self, M):
        return jnp.interp(M, self.m_s, self.dsigmasqdm_s)
    

class SigmaMInterpCube:
    
    def __init__(self, res=500):

        self.name = 'cube'
        self.res = res
        self.m_s = jnp.geomspace(M_MIN, M_MAX, self.res)


    def compute_interps(self):

        ps = NormalizedPowerSpectrum()
        w = CubeWindow()

        class CISigmaSq (CubeIntegrator):
            def integrand(self, kx, ky, kz, R):
                k = jnp.sqrt(kx**2 + ky**2 + kz**2)
                return ps.k2P(k) / (4*jnp.pi*k**2) * w.W(kx, ky, kz, R)**2
        ci_sigmasq = CISigmaSq(ps.k_s, res=1024)

        def sigma_z0_raw(M):
            R = w.MtoR(M)
            return jnp.sqrt(ci_sigmasq.integral(R=R))

        class CIDSigmaSqDR (CubeIntegrator):
            def integrand(self, kx, ky, kz, R):
                k = jnp.sqrt(kx**2 + ky**2 + kz**2)
                return ps.k2P(k) / (4*jnp.pi*k**2) * 2 * w.W(kx, ky, kz, R) * w.dWdR(kx, ky, kz, R)
            
        ci_dsigmasqdm = CIDSigmaSqDR(ps.k_s, res=1024)

        def dsigmasqdm_z0_raw(M):
            R = w.MtoR(M)
            dRdM = R / (3*M)
            dsigmasqdR = ci_dsigmasqdm.integral(R=R)
            return dsigmasqdR * dRdM

        self.sigma_s = []
        for m in tqdm(self.m_s):
            self.sigma_s.append(sigma_z0_raw(m))
        self.sigma_s = jnp.asarray(self.sigma_s)

        self.dsigmasqdm_s = []
        for m in tqdm(self.m_s):
            self.dsigmasqdm_s.append(dsigmasqdm_z0_raw(m))
        self.dsigmasqdm_s = jnp.asarray(self.dsigmasqdm_s)


    def save(self, path):
        data = {
            "res": self.res,
            "m_s": self.m_s,
            "sigma_s": self.sigma_s,
            "dsigmasqdm_s": self.dsigmasqdm_s
        }
        save_h5_dict(path, data)

    def load(self, path):
        data = load_h5_dict(path)
        self.res = data["res"]
        self.m_s = jnp.asarray(data["m_s"])
        self.sigma_s = jnp.asarray(data["sigma_s"])
        self.dsigmasqdm_s = jnp.asarray(data["dsigmasqdm_s"])

    def sigma_z0(self, M):
        return jnp.interp(M, self.m_s, self.sigma_s)
    
    def dsigmasqdm_z0(self, M):
        return jnp.interp(M, self.m_s, self.dsigmasqdm_s)
    

#===== Halo mass function =====

class HMFEvaluator:

    def __init__(self, smi):
        """Evaluate the halo mass function given sigma-m and dsigmasqdm interpolators.
        
        Args:
            smi (SigmaMInterp): Interpolator for sigma and dsigmasqdm.
        """
        self.smi = smi
        self.dicke_z_s = jnp.linspace(0, 100, 1000)
        self.dicke_omega_s = cosmo.Om(self.dicke_z_s)
        self.DELTA_C = 1.68

    def dicke(self, z):
        """Taken exactly from 21cmFAST."""
        omegaM_z = jnp.interp(z, self.dicke_z_s, self.dicke_omega_s)
        dicke_z = 2.5 * omegaM_z / (1/70 + omegaM_z * (209-omegaM_z) / 140 + omegaM_z**(4/7))
        dicke_0 = 2.5 * cosmo.Om0 / (1/70 + cosmo.Om0 * (209-cosmo.Om0) / 140 + cosmo.Om0**(4/7))
        return dicke_z / (dicke_0 * (1+z))

    # @jax.jit
    def dNdM_ST(self, M, z):
        """Sheth-Tormen mass function exactly taken from 21cmFAST,
        which is why the conventions are a little bit weird. This
        matches HMF precisely if the Sheth-Tormen parameteres are
        tuned to match those used by 21cmFAST.
        """
        SHETH_a = 0.73
        SHETH_A = 0.353
        SHETH_p = 0.175

        growthf = self.dicke(z)
        sigma = self.smi.sigma_z0(M)
        dsigmadm = self.smi.dsigmasqdm_z0(M)
        sigma = sigma * growthf
        dsigmadm = dsigmadm * (growthf**2/(2*sigma))
        nuhat = jnp.sqrt(SHETH_a) * self.DELTA_C / sigma
        return - (RHO_M/M) * (dsigmadm/sigma) * jnp.sqrt(2/jnp.pi) * SHETH_A * (1 + nuhat**(-2*SHETH_p)) * nuhat * jnp.exp(-nuhat**2/2)

    # @jax.jit
    def dNdM(self, M, z):
        """Press-Schechter mass function exactly from 21cmFAST,
        which is why the conventions are a little bit weird. This
        precisely matches the HMF module.
        """
        growthf = self.dicke(z)
        sigma = self.smi.sigma_z0(M)
        dsigmadm = self.smi.dsigmasqdm_z0(M)
        sigma = sigma * growthf
        dsigmadm = dsigmadm * (growthf**2/(2*sigma))
        return - (RHO_M/M) * (dsigmadm/sigma) * jnp.sqrt(2/jnp.pi) * (self.DELTA_C/sigma) * jnp.exp(-(self.DELTA_C**2)/(2*sigma**2))
    
    # @jax.jit
    def dNdM_Conditional(self, MR, deltaR, z):
        """Conditional Press-Schechter mass function.
        See Eq. 7.81 in Galaxy.
        """
        
        delta2 = deltaR / self.dicke(z)
        delta1 = self.DELTA_C / self.dicke(z)
        
        S1 = self.smi.sigma_z0(self.smi.m_s)**2
        S2 = self.smi.sigma_z0(MR)**2
        
        dS1_dM1 = self.smi.dsigmasqdm_z0(self.smi.m_s)

        n12 = (delta1 - delta2) / jnp.sqrt(S1 - S2)
        differential = - self.smi.m_s * (delta1-delta2) / 2 / n12 / jnp.sqrt(S1-S2)**3 * dS1_dM1
        differential *= (S1 > S2)
        
        fPS = jnp.sqrt(2/jnp.pi) * n12 * jnp.exp(- n12**2 / 2)
        
        return jnp.where(S1 > S2, 1 / self.smi.m_s**2 * fPS * jnp.abs(differential), jnp.nan) * RHO_M