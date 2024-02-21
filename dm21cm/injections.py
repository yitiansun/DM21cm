"""Classes for handling injections."""

import os
import sys

import numpy as np
import jax.numpy as jnp

sys.path.append(os.environ['DM21CM_DIR'])
import dm21cm.physics as phys

sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec import pppc

class Injection: # Abstract template class

    def __init__(self):
        pass

    def set_binning(self, abscs):
        """Set injection spectra according to binning chosen in evolve.
        Called by evolve during initialization.

        Args:
            abscs (dict): Abscissas/binning for the run.
        """
        pass

    def is_injecting_elec(self):
        """Whether DM is injecting electron/positron. Used by evolve."""
        pass

    def inj_phot_spec_box(self, z_start, dt, **kwargs):
        """Injected photon spectrum and weight box starting from redshift z_start, for duration dt.
        Called by transfer functions wrapper every redshift step.

        Args:
            z_start (float): Starting redshift of the redshift step of injection.
            dt (float): Duration of the redshift step in [s]. Specified in evolve to avoid inconsistent
                dt calculations.

        Returns:
            Spectrum: Injected photon spectrum [ph / Bavg] (number of photons per average Baryon).
            ndarray: Injection weight box of the above spectrum [1].

        Note:
            The output injection is spec \otimes weight_box, with spec carrying the units.
        """
        pass

    def inj_elec_spec_box(self, z_start, dt, **kwargs):
        """Injected electron spectrum and weight box starting from redshift z_start, for duration dt.
        Called by transfer functions wrapper every redshift step. See inj_phot_spec_box for details.
        """
        pass

    def dE_inj_per_Bavg(self):
        """Total energy injected in redshift step per average Baryon [eV/Bavg].
        Called by evolve for recording."""
        pass

    def __eq__(self, other):
        """Equality comparison. Used in darkhistory wrapper to check if cached solution has the correct injection."""
        pass


class DMDecayInjection (Injection):
    """Dark matter decay injection object.
    
    Args:
        primary (str): Primary injection channel. See darkhistory.pppc.get_pppc_spec
        m_DM (float): DM mass in [eV].
        lifetime (float, optional): Decay lifetime in [s].
    """

    def __init__(self, primary=..., m_DM=..., lifetime=...):
        self.mode = 'DM decay'
        self.primary = primary
        self.m_DM = m_DM
        self.lifetime = lifetime

    def set_binning(self, abscs):
        self.inj_phot_spec = pppc.get_pppc_spec(
            self.m_DM, abscs['photE'], self.primary, 'phot', decay=True
        ) # per injection event
        self.inj_elec_spec = pppc.get_pppc_spec(
            self.m_DM, abscs['elecEk'], self.primary, 'elec', decay=True
        ) # per injection event

    def is_injecting_elec(self):
        return not np.allclose(self.inj_elec_spec.N, 0.)

    def inj_phot_spec_box(self, z_start, dt, delta_plus_one_box=..., **kwargs):
        nBavg = phys.n_B * (1+z_start)**3 # [Bavg / (physical cm)^3]
        rho_DM_box = delta_plus_one_box * phys.rho_DM * (1+z_start)**3 # [eV / (physical cm)^3]
        inj_rate_box = (rho_DM_box/self.m_DM) / self.lifetime # [inj / (physical cm)^3 s]
        self.inj_per_Bavg_box = inj_rate_box * dt / nBavg # [inj / Bavg]

        box_avg = float(jnp.mean(self.inj_per_Bavg_box)) # [inj / Bavg]
        return self.inj_phot_spec * box_avg, self.inj_per_Bavg_box / box_avg # [1/Bvg], [1]

    def inj_elec_spec_box(self, z_start, dt, **kwargs):
        # must reuse self.inj_per_Bavg_box
        box_avg = float(jnp.mean(self.inj_per_Bavg_box)) # [inj / Bavg]
        return self.inj_elec_spec * box_avg, self.inj_per_Bavg_box / box_avg # [1/Bvg], [1]
    
    @property
    def dE_inj_per_Bavg(self):
        return jnp.mean(self.inj_per_Bavg_box) * self.m_DM
    
    def __eq__(self, other):
        return (
            self.mode == other.mode and
            self.primary == other.primary and
            self.m_DM == other.m_DM and
            self.lifetime == other.lifetime
        )