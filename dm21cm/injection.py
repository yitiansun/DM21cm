"""Functions handling dark matter energy injection."""

import os, sys
import pandas as pd

import numpy as np
import jax.numpy as jnp

from scipy import interpolate

if os.environ['USER'] == 'yitians' and 'submit' in os.uname().nodename:
    os.environ['DM21CM_DATA_DIR'] = '/data/submit/yitians/dm21cm/DM21cm'
    os.environ['DH_DIR'] = '/work/submit/yitians/darkhistory/DarkHistory'

sys.path.append('..')
sys.path.append(os.environ['DH_DIR'])
    
from darkhistory.spec.spectrum import Spectrum
import darkhistory.spec.spectools as spectools
from darkhistory.spec.pppc import get_pppc_spec

import dm21cm.physics as phys
from dm21cm.common import abscs_nBs_test_2 as abscs
from dm21cm.interpolators import BatchInterpolator5D, BatchInterpolator4D


# Global data structures
global_phot_dep_tf = None
global_elec_dep_tf = None


####################
## DMParams
class DMParams:
    """Dark matter parameters.
    
    Parameters
    ----------
    mode : {'swave', 'decay'}
        Type of injection.
    channel : see darkhistory.pppc.get_pppc_spec
        Injection channel.
    m_DM : float
        DM mass in eV.
    sigmav : float, optional
        Annihilation cross section in cm\ :sup:`3`\ s\ :sup:`-1`\ .
    lifetime : float, optional
        Decay lifetime in s.
    """
    
    def __init__(self, mode, primary, m_DM, sigmav=None, lifetime=None):
        
        if mode == 'swave':
            if sigmav is None:
                raise ValueError('must initialize sigmav.')
        elif mode == 'decay':
            if lifetime is None:
                raise ValueError('must initialize lifetime.')
        else:
            raise NotImplementedError(mode)
        
        self.mode = mode
        self.primary = primary
        self.m_DM = m_DM
        self.sigmav = sigmav
        self.lifetime = lifetime
        
        self.inj_phot_spec = get_pppc_spec(
            self.m_DM, abscs['photE'], self.primary, 'phot',
            decay=(self.mode=='decay')
        ) # injected spectrum per injection event
        self.inj_elec_spec = get_pppc_spec(
            self.m_DM, abscs['elecEk'], self.primary, 'elec',
            decay=(self.mode=='decay')
        ) # injected spectrum per injection event
        
        toteng = self.inj_phot_spec.toteng() + self.inj_elec_spec.toteng()
        self.inj_phot_spec_eng_normalized = self.inj_phot_spec / toteng
        self.inj_elec_spec_eng_normalized = self.inj_elec_spec / toteng
        
        
    def __repr__(self):
        
        return f"DMParams(mode={self.mode}, primary={self.primary}, "\
    f"m_DM={self.m_DM:.4e}, sigmav={self.sigmav}, lifetime={self.lifetime})"
        


####################
## input boxs

def get_input_boxs(delta_box, x_e_box, z_prev, z, dm_params,
                   f_scheme='DH', struct_boost_model='erfc 1e-3'):
    
    # assuming delta = delta_B = delta_DM
    rho_DM_box = phys.rho_DM * (1+z)**3 * (1 + delta_box) # [eV cm^-3]
    n_B_box = phys.n_B * (1+z)**3 * (1 + delta_box) # [cm^-3]

    dE_inj_dVdt_box = phys.inj_rate_box(rho_DM_box, dm_params) # [eV cm^-3 s^-1]
    if dm_params.mode == 'swave':
        dE_inj_dVdt_box *= phys.struct_boost_func(model=struct_boost_model)(1+z)

    dt = phys.dt_between_z(z_prev, z)
    dE_inj_per_B_box = dE_inj_dVdt_box * dt / n_B_box # [eV per B]
    
    # get f boxs
    if f_scheme == 'DH':
        f_boxs = get_DH_f_boxs(delta_box, x_e_box, z, dm_params)
    elif f_scheme == 'EVFY':
        f_boxs = get_EVFY_f_boxs(x_e_box, z)
    elif f_scheme == 'CK':
        f_boxs = get_CK_f_boxs(x_e_box)
    elif f_scheme == 'SPF':
        f_boxs = get_SPF_f_boxs(x_e_box, z, dm_params)
    else:
        raise ValueError('unknown f_scheme.')
        
    n_lya = dE_inj_dVdt_box * dt * f_boxs['exc'] / phys.lya_eng # [lya photon # cm^-3]
    dnu_lya = (phys.rydberg - phys.lya_eng) / (2*np.pi*phys.hbar) # [Hz^-1]
    J_lya = n_lya * phys.c / (4*np.pi) / dnu_lya # [lyaph# cm^-2 s^-1 sr^-1 Hz^-1]
    
    return {
        'heat' : np.array(2 / (3*phys.kB*(1+x_e_box)) * dE_inj_per_B_box * f_boxs['heat']), # [K]
        'ion'  : np.array(dE_inj_per_B_box * f_boxs['ion'] / phys.rydberg), # [1 per B]
        'exc'  : np.array(J_lya), # [lyaph# cm^-2 s^-1 sr^-1 Hz^-1]
        'dE_inj_per_B_mean' : np.mean(dE_inj_per_B_box), # [eV per B]
        'f_heat_mean' : np.mean(f_boxs['heat']),
        'f_ion_mean' : np.mean(f_boxs['ion']),
        'f_exc_mean' : np.mean(f_boxs['exc']),
    }
        
        
####################
## f boxs
        
def get_DH_f_boxs(delta_box, x_e_box, z, dm_params):
    
    global global_phot_dep_tf, global_elec_dep_tf
    
    if global_phot_dep_tf is None:
        global_phot_dep_tf = BatchInterpolator5D(
            os.environ['DM21CM_DATA_DIR'] + \
            '/transferfunctions/nBs_test_2/phot_dep_dlnz4.879E-2_renxo_ad.p'
        )
    if global_elec_dep_tf is None:
        global_elec_dep_tf = BatchInterpolator4D(
            os.environ['DM21CM_DATA_DIR'] + \
            '/transferfunctions/nBs_test_2/elec_dep_dlnz4.879E-2_rexo_ad.p'
        )
    
    # add input checks
    DIM = delta_box.shape[0]
    
    nBs_in = jnp.array(1+delta_box).flatten()
    x_in = jnp.array(x_e_box).flatten()
    
    phot_f_boxs = global_phot_dep_tf(
        1+z,
        dm_params.inj_phot_spec_eng_normalized.N,
        nBs_in,
        x_in,
        out_of_bounds_action='clip'
    ).reshape(DIM, DIM, DIM, 5)
    
    elec_f_boxs = global_elec_dep_tf(
        1+z,
        dm_params.inj_elec_spec_eng_normalized.N,
        x_in,
        out_of_bounds_action='clip'
    ).reshape(DIM, DIM, DIM, 5)
    
    out_absc = np.array(global_phot_dep_tf.abscs['out'])
    
    f_boxs = {}
    i = np.where(out_absc=='heat')[0][0]
    f_boxs['heat'] = phot_f_boxs[:,:,:,i] + elec_f_boxs[:,:,:,i]
    
    i = np.where(out_absc=='H ion')[0][0]
    f_boxs['ion'] = phot_f_boxs[:,:,:,i] + elec_f_boxs[:,:,:,i]
    i = np.where(out_absc=='He ion')[0][0]
    f_boxs['ion'] = phot_f_boxs[:,:,:,i] + elec_f_boxs[:,:,:,i]
    
    i = np.where(out_absc=='exc')[0][0]
    f_boxs['exc'] = phot_f_boxs[:,:,:,i] + elec_f_boxs[:,:,:,i]
    
    return f_boxs


def get_SPF_f_boxs(x_e_box, z, dm_params):
    """Slatyer Padmanabhan Finkbeiner 0906.1197
    (Evoli Mesinger Ferrara 1408.1109 is supposedly close to this)
    Data only covers 1+z from 10 to 1000. Extrapolated with quadratic polynomial
    to 1+z=7.
    """
    
    if not (dm_params.mode == 'swave' and dm_params.primary == 'mu' and dm_params.m_DM == 1e10):
        raise NotImplementedError(dm_params)
        
    rs = 1+z
    
    if not (rs <= 1000. and rs >= 6.9):
        raise ValueError('rs out of bounds.')
    
    data = np.loadtxt('../data/SPF_f_mumu10GeV.txt', skiprows=1, delimiter=',', unpack=True)
    if rs > 10:
        f = interpolate.interp1d(data[0], data[1])(rs)
    else:
        pfits = np.polyfit(np.log(data[0][:11]), np.log(data[1][:11]), deg=2)
        f = np.exp(np.polyval(pfits, np.log(rs)))
    
    return {
        'heat' : np.full_like(x_e_box, f/3.),
        'ion'  : np.full_like(x_e_box, f/3.),
        'exc'  : np.full_like(x_e_box, f/3.),
    }



def get_EVFY_f_boxs(x_e_box, z):
    """Evoli Mesinger Ferrara 1408.1109 uses
    Evoli Valdes Ferrara Yoshida
    https://academic.oup.com/mnras/article/422/1/420/1022144"""
    
    EVFY_f_data = pd.read_csv('../data/EVFY_f_mumu1TeV.txt', sep=' ', index_col=0)
        
    logz = np.log10(z)
    # heat
    fd = EVFY_f_data.loc['f_h']
    Az = fd['A0'] + logz * fd['A1'] + logz**2 * fd['A2']
    Bz = fd['B0'] + logz * fd['B1'] + logz**2 * fd['B2']
    Cz = fd['C0'] + logz * fd['C1'] + logz**2 * fd['C2']
    f_heat = 10**Az * (1 - Cz * x_e_box**Bz)
    
    # H ion
    fd = global_EVFY_f_data.loc['f_iH']
    Az = fd['A0'] + logz * fd['A1'] + logz**2 * fd['A2']
    Bz = fd['B0'] + logz * fd['B1'] + logz**2 * fd['B2']
    Cz = fd['C0'] + logz * fd['C1'] + logz**2 * fd['C2']
    f_H_ion = 10**Az * (1 - x_e_box**Bz)**Cz
    
    # He ion
    fd = global_EVFY_f_data.loc['f_iHe']
    Az = fd['A0'] + logz * fd['A1'] + logz**2 * fd['A2']
    Bz = fd['B0'] + logz * fd['B1'] + logz**2 * fd['B2']
    Cz = fd['C0'] + logz * fd['C1'] + logz**2 * fd['C2']
    f_He_ion = 10**Az * (1 - x_e_box**Bz)**Cz
    
    # exc
    fd = global_EVFY_f_data.loc['f_a']
    Az = fd['A0'] + logz * fd['A1'] + logz**2 * fd['A2']
    Bz = fd['B0'] + logz * fd['B1'] + logz**2 * fd['B2']
    Cz = fd['C0'] + logz * fd['C1'] + logz**2 * fd['C2']
    f_exc = 10**Az * (1 - x_e_box**Bz)**Cz
    
    return {
        'heat' : f_heat,
        'ion'  : f_H_ion + f_He_ion,
        'exc'  : f_exc,
    }


def get_CK_f_boxs(x_e_box):
    """Chen kamionkowski 0310473"""
    
    return {
        'heat' : (1 + 2 * x_e_box) / 3,
        'ion'  : (1 - x_e_box) / 3,
        'exc'  : (1 - x_e_box) / 3,
    }