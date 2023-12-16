import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.append("..")
import dm21cm.physics as phys

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec import pppc
from darkhistory.spec.spectrum import Spectrum

@dataclass
class DMParams:
    """Dark matter parameters.
    
    Args:
        mode {'swave', 'decay'}: Type of injection.
        primary (str): Primary injection channel. See darkhistory.pppc.get_pppc_spec
        m_DM (float): DM mass in eV.
        sigmav (float, optional): Annihilation cross section in cm\ :sup:`3`\ s\ :sup:`-1`\ .
        lifetime (float, optional): Decay lifetime in s.
        struct_boost_model {'erfc 1e-3', 'erfc 1e-6', 'erfc 1e-9'}: Model for structure boost factor. Default is 'erfc 1e-3'.
        
    Attributes:
        inj_phot_spec (Spectrum): Injected photon spectrum per injection event.
        inj_elec_spec (Spectrum): Injected electron positron spectrum per injection event.
        eng_per_inj (float): Injected energy per injection event.
    """
    mode: str
    primary: str
    m_DM: float
    sigmav: float = None
    lifetime: float = None
    struct_boost_model: str = 'erfc 1e-3'
    inj_phot_spec: Spectrum = field(init=False, repr=False, compare=False, default=None)
    inj_elec_spec: Spectrum = field(init=False, repr=False, compare=False, default=None)
    eng_per_inj: float = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self):
        self.eng_per_inj = self.m_DM if self.mode=='decay' else 2 * self.m_DM
        if self.mode == 'swave' and self.sigmav is None:
            raise ValueError('sigmav must be specified for swave injection.')
        if self.mode == 'decay' and self.lifetime is None:
            raise ValueError('lifetime must be specified for decay injection.')

    def set_inj_specs(self, abscs):
        """Set injection spectra according to absicissas."""
        self.inj_phot_spec = pppc.get_pppc_spec(
            self.m_DM, abscs['photE'], self.primary, 'phot',
            decay=(self.mode=='decay')
        )
        self.inj_elec_spec = pppc.get_pppc_spec(
            self.m_DM, abscs['elecEk'], self.primary, 'elec',
            decay=(self.mode=='decay')
        )

    @property
    def is_injecting_elec(self):
        """Whether DM is injecting electron/positron."""
        return not np.allclose(self.inj_elec_spec.N, 0.)

    def struct_boost(self, rs):
        """Structure boost factor as a function of redshift."""
        if self.mode == 'swave':
            return phys.struct_boost_func(model=self.struct_boost_model)(rs)
        elif self.mode == 'decay':
            return 1.
        else:
            raise NotImplementedError(self.mode)