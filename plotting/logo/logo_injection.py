import os
import sys

from PIL import Image
import numpy as np

WDIR = os.environ['DM21CM_DIR']
sys.path.append(WDIR)
import dm21cm.physics as phys
from dm21cm.injections.base import Injection

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec import pppc

logo_path_default = WDIR + '/plotting/logo/source.png'


class LogoInjection (Injection):
    """Inject DM21cm logo as X-ray photons (1keV DM decay to two photons)."""

    def __init__(self, box_dim):
        self.mode = 'Logo'
        self.box_dim = box_dim
        self.load()

        self.m_DM = 1e3 # [eV]
        self.lifetime = 3e25 # [s]

    def set_binning(self, abscs):
        self.phot_spec_per_inj = pppc.get_pppc_spec(
            1e3, abscs['photE'], 'phot_delta', 'phot', decay=True
        ) # [phot / inj]

    def load(self, logo_path=logo_path_default):
        logo = Image.open(logo_path).convert("L")
        logo = np.array(logo.resize((self.box_dim, self.box_dim)))
        logo = np.flipud(1 - logo / 255)
        logo /= np.mean(logo)
        self.logo = np.repeat([logo], self.box_dim, axis=0)

    def inj_multiplier(self, z):
        m = (z/20) ** -9
        m = np.clip(m, 0, 1e4)
        m = np.where(m < 10, np.log(np.exp(m) + np.exp(1)), m)
        return self.logo * m

    def is_injecting_elec(self):
        return False
    
    def get_config(self):
        return {
            'mode': self.mode
        }
    
    #===== injections =====
    def inj_rate(self, z):
        rho_DM = phys.rho_DM * (1+z)**3 # [eV / pcm^3]
        return float((rho_DM/self.m_DM) / self.lifetime) # [inj / pcm^3 s]
    
    def inj_power(self, z):
        return self.inj_rate(z) * self.m_DM # [eV / pcm^3 s]
    
    def inj_phot_spec(self, z, **kwargs):
        return self.phot_spec_per_inj * self.inj_rate(z) # [phot / pcm^3 s]
    
    def inj_phot_spec_box(self, z, delta_plus_one_box=..., **kwargs):
        return self.inj_phot_spec(z), delta_plus_one_box * self.inj_multiplier(z) # [phot / pcm^3 s], [1]