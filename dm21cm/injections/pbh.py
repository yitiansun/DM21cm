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
from dm21cm.interpolators import interp1d, interp2d_vmap, bound_action
import dm21cm.physics as phys
from dm21cm.utils import load_h5_dict, abscs

data_dir = f'{WDIR}/data'


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
        try:
            self.data = load_h5_dict(f'{data_dir}/pbh-hr/pbh_logm{np.log10(m_PBH):.3f}.h5')
        except FileNotFoundError:
            raise FileNotFoundError(f'PBH data for log10(m_PBH/g)={np.log10(m_PBH):.3f} not found.')
        
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

    def set_binning(self, abscs):
        self.abscs = abscs

    def is_injecting_elec(self):
        return True
    
    def get_config(self):
        return {
            'mode': self.mode,
            'm_PBH': self.m_PBH,
            'f_PBH': self.f_PBH
        }
    
    #===== final injection modification =====
    # def init_final_inj(self, z_inj_s):
    #     """Initialize final stage injection parameters if PBH has evaporated by z_end.
    #     Will be called for DarkHistory and DM21cm.

    #     Args:
    #         z_inj_s (array): List of (decreasing) redshifts at which injection happens, plus a final boundary z.

    #     Set the following attributes:
    #         z_inj_s (array): As above.
    #         evaporated_by_end_rs (bool): True if PBH has evaporated by end_rs.
    #         if evaporated_by_end_rs is True:
    #             z_final_inj (float): Redshift of the final injection step.
    #             final_inj_multiplier (float): Injection energy multiplier of the final injection step.
    #             phot_final_inj_shape (Spectrum): Injection spectral shape of photons at the final injection step.
    #             elec_final_inj_shape (Spectrum): Injection spectral shape of electrons at the final injection step.
    #     """
    #     # Get dM's
    #     dM_s = []
    #     for i, z in enumerate(z_inj_s[:-1]):
    #         t = phys.t_z(z)
    #         t_next = phys.t_z(z_inj_s[i+1])
    #         dM = self.dMdt_t(t) * (t_next - t)
    #         dM_s.append(dM) # dM's length will be one less than z_inj_s

    #     # Test if BH has evaporated by end_rs
    #     self.z_inj_s = z_inj_s
    #     if dM_s[-1] != 0.: # BH has not evaporated
    #         self.evaporated_by_end = False
    #         return
    #     self.evaporated_by_end = True
    #     self.i_final_inj = np.nonzero(dM_s)[0][-1] # there is a danger of emission spec mismatch with dMdt
    #     self.z_final_inj = z_inj_s[self.i_final_inj]

    #     # Final step injection multiplier
    #     dM_total = self.data['M0'] - self.M_t(phys.t_z(z_inj_s[0])) # [g]
    #     dM_actual = np.sum(dM_s) # [g]
    #     dM_extra = np.max(dM_total - dM_actual, 0) # [g]
    #     self.final_inj_multiplier = (dM_extra + dM_s[self.i_final_inj]) / dM_s[self.i_final_inj]

    #     # Final step injection spectral shape
    #     t_final_start = phys.t_z(np.sqrt((1+self.z_final_inj) * (1+self.z_inj_s[self.i_final_inj-1])) - 1) # [s]
    #     phot_dNdE = self.data['phot dNdEdt'][0] * 0.
    #     elec_dNdE = self.data['elec dNdEdt'][0] * 0.
    #     for i, t in enumerate(self.data['t']):
    #         if t < t_final_start:
    #             continue
    #         dt = t - self.data['t'][i-1]
    #         phot_dNdE += self.data['phot dNdEdt'][i] * dt
    #         elec_dNdE += self.data['elec dNdEdt'][i] * dt
    #     self.phot_final_inj_shape = Spectrum(self.abscs['photE'], phot_dNdE, spec_type='dNdE') # normalization does not matter
    #     self.elec_final_inj_shape = Spectrum(self.abscs['elecEk'], elec_dNdE, spec_type='dNdE')
    

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

        self.halo_data = load_h5_dict(f"{os.environ['DM21CM_DATA_DIR']}/pbhacc_halo_hmf_summed_rate_{self.model}.h5")
        self.cosmo_data = load_h5_dict(f"{os.environ['DM21CM_DATA_DIR']}/pbhacc_cosmo_rate_{self.model}.h5")
        
        if m_PBH not in self.halo_data['mPBH']:
            raise ValueError(f'PBH data for m_PBH={m_PBH} not found.')
        self.i_mPBH = np.where(self.halo_data['mPBH'] == m_PBH)[0][0]

        self.z_s = self.halo_data['z']
        self.zfull_s = self.halo_data['zfull']
        self.d_s = self.halo_data['d']
        self.cinf_s = self.cosmo_data['cinf'] # [km/s]
        self.halo_ps_cond = self.halo_data['ps_cond'][self.i_mPBH] # [eV / s / cfcm^3]
        self.halo_ps = self.halo_data['ps'][self.i_mPBH]
        self.halo_st = self.halo_data['st'][self.i_mPBH]
        self.cosmo_ps_cond = self.cosmo_data['ps_cond'][self.i_mPBH]
        self.cosmo_ps = self.cosmo_data['ps'][self.i_mPBH]
        self.cosmo_st = self.cosmo_data['st'][self.i_mPBH]

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

    def set_binning(self, abscs):
        self.abscs = abscs

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
            T_k (float): Gas temperature [eV].
            x_e (float): Ionization fraction [1].
        """
        return np.sqrt(5/3 * (1 + x_e) * T_k * u.eV / c.m_p).to(u.km/u.s).value
    
    def cinf_std(self, z):
        """Standard ambient sound speed [km/s]."""
        T_k = dh_phys.Tm_std(1+z) # [eV]
        x_e = dh_phys.xHII_std(1+z) # [1]
        return self.cinf(T_k, x_e)
    
    def inj_power_std(self, z):
        T_k = dh_phys.Tm_std(1+z) # [eV]
        x_e = dh_phys.xHII_std(1+z) # [1]
        return self.inj_power(z, state=dict(Tm=T_k, xHII=x_e))

    
    #===== injections =====
    def inj_rate(self, z_start, z_end=None, **kwargs):
        return self.inj_rate_ref # [inj / pcm^3 s]
    
    def inj_power(self, z_start, z_end=None, state=None, **kwargs):
        z_in = bound_action(z_start, self.zfull_s, 'clip')
        cinf_in = self.cinf(state['Tm'], state['xHII']) if state is not None else self.cinf_std(z_start)
        cinf_in = bound_action(cinf_in, self.cinf_s, 'clip') # [km/s]
        halo_power = interp1d(self.halo_st, self.zfull_s, z_in) # [eV / s / cfcm^3]
        cosmo_power = interp1d(interp1d(self.cosmo_st, self.zfull_s, z_in), self.cinf_s, cinf_in) # [eV / s / cfcm^3]
        return self.f_PBH * (halo_power + cosmo_power) * (1 + z_start)**3 # [eV / pcm^3 s]
    
    def inj_phot_spec(self, z_start, z_end=None, state=None, **kwargs):
        return self.phot_spec * float(self.inj_power(z_start, state=state)) # [phot/eV / pcm^3 s]
    
    def inj_elec_spec(self, z_start, z_end=None, **kwargs):
        return self.zero_elec_spec # [elec/eV / pcm^3 s]
    
    def inj_phot_spec_box(self, z_start, z_end=None, delta_plus_one_box=None, **kwargs):
        z_in = bound_action(z_start, self.z_s, 'raise') # can only access from DM21cm
        cinf_in = bound_action(self.cinf_std(z_start), self.cinf_s, 'clip') # [km/s] | for cosmo PBH, assume standard cosmology
        d_box_in = bound_action(delta_plus_one_box - 1, self.d_s, 'clip')

        # table units: [eV / s / cfcm^3]
        halo_ps_cond_d = interp1d(self.halo_ps_cond, self.z_s, z_in) # shape=(d,)
        halo_ps_d = interp1d(self.halo_ps, self.z_s, z_in) # shape=()
        halo_st_d = interp1d(self.halo_st, self.z_s, z_in) # shape=()
        halo_power = interp1d(halo_ps_cond_d, self.d_s, d_box_in) # shape=box
        halo_power *= halo_st_d / halo_ps_d

        cosmo_ps_cond_d = interp1d(interp1d(self.cosmo_ps_cond, self.zfull_s, z_in), self.cinf_s, cinf_in) # shape=(d,)
        cosmo_ps_d = interp1d(interp1d(self.cosmo_ps, self.zfull_s, z_in), self.cinf_s, cinf_in) # shape=()
        cosmo_st_d = interp1d(interp1d(self.cosmo_st, self.zfull_s, z_in), self.cinf_s, cinf_in) # shape=()
        cosmo_power = interp1d(cosmo_ps_cond_d, self.d_s, d_box_in) # shape=box
        cosmo_power *= cosmo_st_d / cosmo_ps_d

        power_box = (halo_power + cosmo_power) * self.f_PBH * (1 + z_start)**3 # [eV / pcm^3 s]
        power_mean = jnp.mean(power_box)
        return self.phot_spec * float(power_mean), power_box / power_mean # [phot/eV / pcm^3 s], [1]

    # def inj_phot_spec_box(self, z_start, z_end=None, delta_plus_one_box=None, Tk_box=None, **kwargs):
    #     d_in = bound_action(delta_plus_one_box.flatten() - 1, self.d_s, 'clip')
    #     T_in = bound_action(Tk_box.flatten(), self.T_s, 'clip')
    #     d_T_in = jnp.stack([d_in, T_in], axis=-1)