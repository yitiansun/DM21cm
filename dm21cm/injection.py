"""Functions handling dark matter energy injection."""

import os, sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import jax.numpy as jnp

if os.environ['USER'] == 'yitians' and 'submit' in os.uname().nodename:
    os.environ['DM21CM_DATA_DIR'] = '/data/submit/yitians/DM21cm'
    os.environ['DH_DIR'] = '/work/submit/yitians/darkhistory/DarkHistory'

sys.path.append('..')
sys.path.append(os.environ['DH_DIR'])
    
from darkhistory.spec.spectrum import Spectrum
import darkhistory.spec.spectools as spectools

import dm21cm.physics as phys
from dm21cm.common import abscs_nBs_test_2 as abscs
from dm21cm.interpolators import BatchInterpolator


# Global data structures
global_phot_dep_tf = None


####################
## DMParams
@dataclass
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
    mode : str
    channel : str
    m_DM : float
    sigmav : Optional[float] = None
    lifetime : Optional[float] = None
    
    def __post_init__(self):
        
        if self.mode == 'swave':
            if self.sigmav is None:
                raise ValueError('must initialize sigmav.')
        elif self.mode == 'decay':
            if self.lifetime is None:
                raise ValueError('must initialize lifetime.')
        else:
            raise NotImplementedError(self.mode)


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
    
    global global_phot_dep_tf
    if global_phot_dep_tf is None:
        global_phot_dep_tf = BatchInterpolator(os.environ['DM21CM_DATA_DIR'] + '/transferfunctions/nBs_test_2/phot_dep_dlnz4.879E-2_renxo_ad.p')
    
    DIM = delta_box.shape[0]
    
    nBs_in = jnp.array(1+delta_box).flatten()
    x_in = jnp.array(x_e_box).flatten()
    
    f_boxs = global_phot_dep_tf(
        1+z,
        get_eng_normalized_specs(dm_params),
        nBs_in,
        x_in,
        out_of_bounds_action='clip'
    ).reshape(DIM, DIM, DIM, 5)
    
    out_absc = np.array(global_phot_dep_tf.abscs['out'])
    
    return {
        'heat' : f_boxs[:,:,:, np.where(out_absc=='heat')[0][0]  ],
        'ion'  : f_boxs[:,:,:, np.where(out_absc=='H ion')[0][0] ] \
               + f_boxs[:,:,:, np.where(out_absc=='He ion')[0][0] ],
        'exc'  : f_boxs[:,:,:, np.where(out_absc=='exc')[0][0]   ],
    }


def get_EMF_f_boxs(x_e_box):
    """Evoli Mesinger Ferrara 1408.1109"""
    
    return {
        'heat' : (1 + 2 * x_e_box) / 3,
        'ion'  : (1 - x_e_box) / 3,
        'exc'  : (1 - x_e_box) / 3,
    }


####################
## input spectrum

def get_eng_normalized_specs(dm_params):
    """DM injection spectra normalized such that the total energy of all the
    spectra is 1 (dimensionless).
    """
    
    if dm_params.channel == 'phph':
        
        spec = spectools.rebin_N_arr(
            np.array([1.]),
            np.array([dm_params.m_DM / 2]),
            out_eng=abscs['photE']
        )
        return spec.N/spec.toteng()
    
    else:
        raise NotImplementedError(dm_params.channel)
        