"""Functions handling dark matter energy injection."""

import os, sys

import numpy as np
import jax.numpy as jnp

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
    channel : {'phph'}
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


####################
## input boxs

def get_input_boxs(delta_box, x_e_box, z_prev, z, dm_params, f_scheme='DH'):
    
    # assuming delta = delta_B = delta_DM
    rho_DM_box = phys.rho_DM * (1+z)**3 * (1 + delta_box) # [eV cm^-3]
    n_B_box = phys.n_B * (1+z)**3 * (1 + delta_box) # [cm^-3]

    dE_inj_dVdt_box = phys.inj_rate_box(rho_DM_box, dm_params) # [eV cm^-3 s^-1]
    if dm_params.mode == 'swave':
        dE_inj_dVdt_box *= phys.struct_boost_func(model='erfc')(1+z)

    dt = phys.dt_between_z(z_prev, z)
    dE_inj_per_B_box = dE_inj_dVdt_box * dt / n_B_box # [eV per B]
    
    if f_scheme == 'DH':
        f_boxs = get_DH_f_boxs(delta_box, x_e_box, z, dm_params)
    elif f_scheme == 'EMF':
        f_boxs = get_EMF_f_boxs(x_e_box)
    else:
        raise ValueError('unknown f_scheme.')
    
    return {
        'heat' : np.array(2 / (3*phys.kB*(1+x_e_box)) * dE_inj_per_B_box * f_boxs['heat']), # [K]
        'ion'  : np.array(dE_inj_per_B_box * f_boxs['ion'] / phys.rydberg), # [1 per B]
        #'exc'  : np.array(dE_inj_per_B_box * f_boxs['exc'] / phys.lya_eng), # [??? 1 per B]
        'exc'  : "¯\_(ツ)_/¯",
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


def get_EMF_f_boxs(x_e_box):
    """Evoli Mesinger Ferrara 1408.1109"""
    
    return {
        'heat' : (1 + 2 * x_e_box) / 3,
        'ion'  : (1 - x_e_box) / 3,
        'exc'  : (1 - x_e_box) / 3,
    }