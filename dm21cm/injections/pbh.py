"""Primordial Black Hole (PBH) injection."""

import os
import sys

import numpy as np
from scipy import interpolate
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
import astropy.constants as c

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.spectrum import Spectrum
from darkhistory import physics as dh_phys

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
from dm21cm.injections.base import Injection
from dm21cm.interpolators import interp1d, bound_action, interp3d_vmap
import dm21cm.physics as phys
from dm21cm.utils import load_h5_dict, abscs


class PBHHRInjection (Injection):
    """Primordial Black Hole (PBH) Hawking radiation (HR) injection object. See parent class for details.

    Args:
        m_PBH (float): PBH mass in [g].
        f_PBH (float): PBH fraction of DM.
    """

    def __init__(self, m_PBH, f_PBH=1.):
        self.mode = 'PBH-HR'
        self.m_PBH = m_PBH # [g]
        self.f_PBH = f_PBH
        self.inj_per_sec = 1. # [inj / s] | convention: 1 injection event per second

        self.m_eV = (self.m_PBH * u.g * c.c**2).to(u.eV).value # [eV]
        self.n0_PBH = phys.rho_DM * f_PBH / self.m_eV # [BH / pcm^3] | Present day PBH number density

        #----- Load PBH data -----
        self.data = load_h5_dict(f"{os.environ['DM21CM_DATA_DIR']}/pbhhr.h5")
        self.data = self.data[f'log10m{np.log10(self.m_PBH):.3f}']
        
        i_start = np.where(self.data['t'] > phys.t_z(5e3))[0][0] # 1e4 is the largest z phys.z_t calculates
        i_end = i_start
        while i_end < len(self.data['t']) and self.data['t'][i_end] < phys.t_z(1): # 1e-8 is the smallest z phys.z_t calculates
            if self.data['t'][i_end] == self.data['t'][i_end-1]:
                break
            i_end += 1
        self.t_s = self.data['t'][i_start:i_end] # [s]
        self.M_s = self.data['M'][i_start:i_end] # [g]
        self.phot_dNdEdt_s = self.data['phot dNdEdt'][i_start:i_end] # [phot / eV s BH]
        self.elec_dNdEdt_s = self.data['elec dNdEdt'][i_start:i_end] # [elec / eV s BH]
        
        self.t_edges = (self.t_s[:-1] + self.t_s[1:]) / 2 # [s]
        self.t_edges = np.concatenate((
            [self.t_s[0] - (self.t_edges[0] - self.t_s[0])],
            self.t_edges,
            [self.t_s[-1] + (self.t_s[-1] - self.t_edges[-1])]
        )) # [s]
        self.z_edges = np.array([phys.z_t(t) for t in self.t_edges])

        #----- Interpolations -----
        self.M_t = interpolate.interp1d(self.t_s, self.M_s, bounds_error=False, fill_value=0) # [g]([s])
        zero_spec = self.data['phot dNdEdt'][0] * 0.
        self.phot_dNdEdt_interp = interpolate.interp1d(self.t_s, self.phot_dNdEdt_s, axis=0, bounds_error=False, fill_value=zero_spec) # [phot / eV s BH]
        self.elec_dNdEdt_interp = interpolate.interp1d(self.t_s, self.elec_dNdEdt_s, axis=0, bounds_error=False, fill_value=zero_spec) # [elec / eV s BH]

        dMdt = np.abs(np.gradient(self.M_s, self.t_s)) # [g/s]
        self.dMdt_t = interpolate.interp1d(self.t_s, dMdt, bounds_error=False, fill_value=0) # [g/s]([s])

        self.abscs = abscs

    def is_injecting_elec(self):
        return True
    
    def get_config(self):
        return {
            'mode': self.mode,
            'm_PBH': self.m_PBH,
            'f_PBH': self.f_PBH
        }

    #===== injections =====
    def n_PBH(self, z_start, z_end=None):
        """Mean physical number density of PBHs in [BH / pcm^3]. Includes 'evaporated PBHs'."""
        return self.n0_PBH * (1+z_start)**3 # [BH / pcm^3]

    def inj_rate(self, z_start, z_end=None, **kwargs):
        return self.n_PBH(z_start, z_end=z_end) * self.inj_per_sec # [inj / pcm^3 s]
    
    def inj_power(self, z_start, z_end=None, **kwargs):
        power = self.inj_phot_spec(z_start, z_end=z_end).toteng() + self.inj_elec_spec(z_start, z_end=z_end).toteng()
        return max(1e-100, power) # [eV / pcm^3 s]

    def inj_phot_spec(self, z_start, z_end=None, **kwargs):
        """Photon injection spectrum [phot / pcm^3 (eV) s]."""
        if z_end is None: # instantaneous rate spectrum
            dndEdt = self.phot_dNdEdt_interp(phys.t_z(z_start)) * self.n_PBH(z_start) # [phot / pcm^3 eV s]
            return Spectrum(self.abscs['photE'], dndEdt, spec_type='dNdE')
        
        t_start = phys.t_z(z_start)
        t_end = phys.t_z(z_end)
        dt = t_end - t_start
        interval_inds = (self.t_edges > t_start) & (self.t_edges < t_end)
        t_edges_interval = np.concatenate(( [t_start], self.t_edges[interval_inds], [t_end] ))
        z_edges_interval = np.concatenate(( [z_start], self.z_edges[interval_inds], [z_end] ))
        dndEdt_s = self.phot_dNdEdt_interp(t_edges_interval) * self.n_PBH(z_edges_interval)[:, None] # [phot / pcm^3 eV s]
        dndE = np.trapz(dndEdt_s, x=t_edges_interval, axis=0) # [phot / pcm^3 eV]
        return Spectrum(self.abscs['photE'], dndE / dt, spec_type='dNdE') # [phot / pcm^3 (eV) s]
        
    def inj_elec_spec(self, z_start, z_end=None, **kwargs):
        """Electron injection spectrum [elec / pcm^3 (eV) s]."""
        if z_end is None: # instantaneous rate spectrum
            dndEdt = self.elec_dNdEdt_interp(phys.t_z(z_start)) * self.n_PBH(z_start) # [elec / pcm^3 eV s]
            return Spectrum(self.abscs['elecEk'], dndEdt, spec_type='dNdE')
        
        t_start = phys.t_z(z_start)
        t_end = phys.t_z(z_end)
        dt = t_end - t_start
        interval_inds = (self.t_edges > t_start) & (self.t_edges < t_end)
        t_edges_interval = np.concatenate(( [t_start], self.t_edges[interval_inds], [t_end] ))
        z_edges_interval = np.concatenate(( [z_start], self.z_edges[interval_inds], [z_end] ))
        dndEdt_s = self.elec_dNdEdt_interp(t_edges_interval) * self.n_PBH(z_edges_interval)[:, None] # [elec / pcm^3 eV s]
        dndE = np.trapz(dndEdt_s, x=t_edges_interval, axis=0) # [elec / pcm^3 eV]
        return Spectrum(self.abscs['elecEk'], dndE / dt, spec_type='dNdE') # [elec / pcm^3 (eV) s]
    
    def inj_phot_spec_box(self, z_start, z_end=None, delta_plus_one_box=None, **kwargs):
        return self.inj_phot_spec(z_start, z_end=z_end), delta_plus_one_box # [phot / pcm^3 (eV) s], [1]

    def inj_elec_spec_box(self, z_start, z_end=None, delta_plus_one_box=None, **kwargs):
        return self.inj_elec_spec(z_start, z_end=z_end), delta_plus_one_box # [elec / pcm^3 (eV) s], [1]
    


_CINF_T_K_REF = 1 # [K]
_CINF_X_E_REF = 0 # [1]
_CINF_REF = np.sqrt(5/3 * (1 + _CINF_X_E_REF) * _CINF_T_K_REF * u.K * c.k_B / c.m_p).to(u.km/u.s).value


class PBHAccretionInjection (Injection):
    """Primordial Black Hole (PBH) accretion injection object. See parent class for details.

    Args:
        m_PBH (float): PBH mass in [M_sun].
        f_PBH (float): PBH fraction of DM.
    """

    def __init__(self, model, m_PBH, f_PBH):
        self.mode = 'PBH-Accretion'
        self.model = model
        self.m_PBH = m_PBH
        self.f_PBH = f_PBH
        self.inj_rate_ref = 1. # [inj / pcm^3 s] | dummy injection rate

        prefix = f"{os.environ['DM21CM_DATA_DIR']}/pbhacc_rates/{self.model}/{self.model}_log10m{np.log10(self.m_PBH):.3f}"
        self.halo_data = load_h5_dict(prefix + "_halo.h5")
        self.cosmo_data = load_h5_dict(prefix + "_cosmo.h5")

        self.z_s           = jnp.array(self.halo_data ['z'])
        self.zfull_s       = jnp.array(self.halo_data ['zfull'])
        self.d_s           = jnp.array(self.halo_data ['d'])
        self.cinf_s        = jnp.array(self.cosmo_data['cinf']) # [km/s]
        self.vcb_s         = jnp.array(self.cosmo_data['vcb']) # [km/s]
        self.halo_ps_cond  = jnp.array(self.halo_data ['ps_cond']) # [eV / s / cfcm^3]
        self.halo_ps       = jnp.array(self.halo_data ['ps'])
        self.halo_st       = jnp.array(self.halo_data ['st'])
        self.cosmo_ps_cond = jnp.array(self.cosmo_data['ps_cond'])
        self.cosmo_ps      = jnp.array(self.cosmo_data['ps'])
        self.cosmo_st      = jnp.array(self.cosmo_data['st'])

        self.init_specs()

    def init_specs(self):
        E = abscs['photE'] # [eV]
        self.E_min = (self.m_PBH / 10) ** (-1/2) # [eV]
        self.E_Ts = 2e5 # [eV]
        self.a = 1
        dLdE = (E > self.E_min) * E**(-self.a) * np.exp(- E / self.E_Ts)
        dNdE = dLdE / E
        E_tot = np.trapz(E * dNdE, E)
        self.phot_spec = Spectrum(E, dNdE / E_tot, spec_type='dNdE') # [phot/eV / eV(injected)]
        self.zero_elec_spec = Spectrum(E, 0. * abscs['elecEk'], spec_type='dNdE') # [elec/eV / eV(injected)]

    def is_injecting_elec(self):
        return False
    
    def get_config(self):
        return {
            'mode': self.mode,
            'model': self.model,
            'm_PBH': self.m_PBH,
            'f_PBH': self.f_PBH
        }
    
    #===== physics =====
    def cinf(self, T_k, x_e):
        """Ambient sound speed [km/s]. Used for contributions from unbound PBH.
        
        Args:
            T_k (float or array): Gas temperature [K].
            x_e (float or array): Ionization fraction [1].
        """
        return _CINF_REF * jnp.sqrt((1 + x_e) / (1 + _CINF_X_E_REF) * T_k / _CINF_T_K_REF)
    
    def inj_power_std(self, z):
        T_k = dh_phys.Tm_std(1+z) # [eV]
        x_e = dh_phys.xHII_std(1+z) # [1]
        return self.inj_power(z, state=dict(Tm=T_k, xHII=x_e))

    
    #===== injections =====
    def inj_rate(self, z_start, z_end=None, **kwargs):
        return self.inj_rate_ref # [inj / pcm^3 s]
    
    def inj_power(self, z_start, z_end=None, state=None, debug=None, **kwargs):
        z_in = bound_action(z_start, self.zfull_s, 'clip')
        if state:
            T_k = (state['Tm'] * u.eV / c.k_B).to(u.K).value
            x_e = state['xHII']
        else:
            T_k = (dh_phys.Tm_std(1+z_start) * u.eV / c.k_B).to(u.K).value # [K]
            x_e = dh_phys.xHII_std(1+z_start) # [1]
        cinf_in = self.cinf(T_k, x_e)
        cinf_in = bound_action(cinf_in, self.cinf_s, 'clip') # [km/s]
        halo_power = interp1d(self.halo_st, self.zfull_s, z_in) # [eV / s / cfcm^3]
        cosmo_power = interp1d(interp1d(self.cosmo_st, self.zfull_s, z_in), self.cinf_s, cinf_in) # [eV / s / cfcm^3]
        if debug == 'halo only':
            cosmo_power = 0
        elif debug == 'cosmo only':
            halo_power = 0
        return self.f_PBH * (halo_power + cosmo_power) * (1 + z_start)**3 # [eV / pcm^3 s]
    
    def inj_phot_spec(self, z_start, z_end=None, state=None, **kwargs):
        return self.phot_spec * float(self.inj_power(z_start, state=state)) # [phot/eV / pcm^3 s]
    
    def inj_elec_spec(self, z_start, z_end=None, **kwargs):
        return self.zero_elec_spec # [elec/eV / pcm^3 s]
    
    def inj_phot_spec_box(self, z_start, z_end=None, delta_plus_one_box=None, T_k_box=None, x_e_box=None, vcb_box=None, **kwargs):
        z_in = bound_action(z_start, self.z_s, 'raise') # can only access from DM21cm
        cinf_box_in = bound_action(self.cinf(T_k_box, x_e_box), self.cinf_s, 'clip') # [km/s]
        cinf_avg = jnp.mean(cinf_box_in) # [km/s]
        d_box_in = bound_action(delta_plus_one_box - 1, self.d_s, 'clip')
        vcb_box_in = bound_action(vcb_box, self.vcb_s, 'clip') # [km/s]

        # table units: [eV / s / cfcm^3]
        halo_ps_cond_val = interp1d(interp1d(self.halo_ps_cond, self.z_s, z_in), self.d_s, d_box_in) # shape=box
        halo_ps_val = interp1d(self.halo_ps, self.z_s, z_in) # shape=()
        halo_st_val = interp1d(self.halo_st, self.z_s, z_in) # shape=()
        halo_power = (halo_st_val / halo_ps_val) * halo_ps_cond_val # shape=box

        cosmo_ps_cond_cdv = interp1d(self.cosmo_ps_cond, self.zfull_s, z_in) # shape=(cinf, d, vcb)
        cdv_in = jnp.stack([cinf_box_in.flatten(), d_box_in.flatten(), vcb_box_in.flatten()], axis=-1) # shape=(boxlen, 3)
        cosmo_ps_cond_val = interp3d_vmap(cosmo_ps_cond_cdv, self.cinf_s, self.d_s, self.vcb_s, cdv_in).reshape(cinf_box_in.shape) # shape=box

        cosmo_ps_val = interp1d(interp1d(self.cosmo_ps, self.zfull_s, z_in), self.cinf_s, cinf_avg) # shape=()
        cosmo_st_val = interp1d(interp1d(self.cosmo_st, self.zfull_s, z_in), self.cinf_s, cinf_avg) # shape=()
        cosmo_power = (cosmo_st_val / cosmo_ps_val) * cosmo_ps_cond_val # shape=box

        power_box = (halo_power + cosmo_power) * self.f_PBH * (1 + z_start)**3 # [eV / pcm^3 s]
        power_mean = jnp.mean(power_box)
        return self.phot_spec * float(power_mean), power_box / power_mean # [phot/eV / pcm^3 s], [1]


    # #===== auxiliary functions =====
    # def inj_halo_power(self, z_start, z_end=None, state=None, **kwargs):
    #     z_in = bound_action(z_start, self.zfull_s, 'clip')
    #     cinf_in = self.cinf(state['Tm'], state['xHII']) if state is not None else self.cinf_std(z_start)
    #     cinf_in = bound_action(cinf_in, self.cinf_s, 'clip') # [km/s]
    #     halo_power = interp1d(self.halo_st, self.zfull_s, z_in) # [eV / s / cfcm^3]
    #     # cosmo_power = interp1d(interp1d(self.cosmo_st, self.zfull_s, z_in), self.cinf_s, cinf_in) # [eV / s / cfcm^3]
    #     return self.f_PBH * halo_power * (1 + z_start)**3 # [eV / pcm^3 s]
    
    # def inj_cosmo_power(self, z_start, z_end=None, state=None, **kwargs):
    #     z_in = bound_action(z_start, self.zfull_s, 'clip')
    #     cinf_in = self.cinf(state['Tm'], state['xHII']) if state is not None else self.cinf_std(z_start)
    #     cinf_in = bound_action(cinf_in, self.cinf_s, 'clip') # [km/s]
    #     # halo_power = interp1d(self.halo_st, self.zfull_s, z_in) # [eV / s / cfcm^3]
    #     cosmo_power = interp1d(interp1d(self.cosmo_st, self.zfull_s, z_in), self.cinf_s, cinf_in) # [eV / s / cfcm^3]
    #     return self.f_PBH * cosmo_power * (1 + z_start)**3 # [eV / pcm^3 s]