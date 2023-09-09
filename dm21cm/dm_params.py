import os
import sys
from dataclasses import dataclass

sys.path.append("..")
import dm21cm.physics as phys

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec import pppc

@dataclass
class DMParams:
    """Dark matter parameters.
    
    Args:
        abscs (dict): Abscissas.
        mode {'swave', 'decay'}: Type of injection.
        primary (str): Primary injection channel. See darkhistory.pppc.get_pppc_spec
        m_DM (float): DM mass in eV.
        sigmav (float, optional): Annihilation cross section in cm\ :sup:`3`\ s\ :sup:`-1`\ .
        lifetime (float, optional): Decay lifetime in s.
        struct_boost_model (str, optional): Model for structure boost factor.
        
    Attributes:
        inj_phot_spec (Spectrum): Injected photon spectrum per injection event.
        inj_elec_spec (Spectrum): Injected electron positron spectrum per injection event.
        eng_per_inj (float): Injected energy per injection event.
    """
    abscs: dict
    mode: str
    primary: str
    m_DM: float
    sigmav: float = None
    lifetime: float = None
    struct_boost_model: str = 'erfc 1e-3'

    def __post_init__(self):

        self.inj_phot_spec = pppc.get_pppc_spec(
            self.m_DM, self.abscs['photE'], self.primary, 'phot',
            decay=(self.mode=='decay')
        )
        self.inj_elec_spec = pppc.get_pppc_spec(
            self.m_DM, self.abscs['elecEk'], self.primary, 'elec',
            decay=(self.mode=='decay')
        )
        self.eng_per_inj = self.m_DM if self.mode=='decay' else 2 * self.m_DM

    def struct_boost(self, rs):
        """Structure boost factor as a function of redshift."""
        if self.mode == 'swave':
            return phys.struct_boost_func(model=self.struct_boost_model)(rs)
        elif self.mode == 'decay':
            return 1.
        else:
            raise NotImplementedError(self.mode)