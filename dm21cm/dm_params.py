"""Class for dark matter (DM) related variables."""

import os
import sys

sys.path.append("..")
from dm21cm.utils import load_dict

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.pppc import get_pppc_spec

abscs = load_dict(os.environ['DM21CM_DIR']+'/data/abscissas/abscs_base.h5')


class DMParams:
    """Dark matter parameters.
    
    Args:
        mode {'swave', 'decay'}: Type of injection.
        primary (str): Primary injection channel. See darkhistory.pppc.get_pppc_spec
        m_DM (float): DM mass in eV.
        sigmav (float): Annihilation cross section in cm\ :sup:`3`\ s\ :sup:`-1`\ .
        lifetime (float, optional): Decay lifetime in s.
        
    Attributes:
        inj_phot_spec (Spectrum): Injected photon spectrum per injection event.
        inj_elec_spec (Spectrum): Injected electron positron spectrum per injection event.
        eng_per_inj (float): Injected energy per injection event.
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
        )
        self.inj_elec_spec = get_pppc_spec(
            self.m_DM, abscs['elecEk'], self.primary, 'elec',
            decay=(self.mode=='decay')
        )
        self.eng_per_inj = self.m_DM if self.mode=='decay' else 2 * self.m_DM
        
        
    def __repr__(self):
        
        return f"DMParams(mode={self.mode}, primary={self.primary}, "\
    f"m_DM={self.m_DM:.4e}, sigmav={self.sigmav}, lifetime={self.lifetime})"
