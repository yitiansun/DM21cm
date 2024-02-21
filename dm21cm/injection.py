

class Injection:

    def __init__(self):
        self.is_injecting_elec = ... # Whether DM is injecting electron/positron. Used by evolve.

    def set_binning(self, abscs):
        """Set injection spectra according to binning chosen in evolve.
        Called by evolve during initialization.

        Args:
            abscs (dict): Abscissas/binning for the run.
        """
        pass

    def inj_phot_spec_box(self, z_start, dt, **kwargs):
        """Injected photon spectrum and weight box from redshift z_start to z_end.
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
        return None, None # spectrum, weight_box

    def inj_elec_spec_box(self, z_start, dt, **kwargs):
        """Inject electron spectrum and weight box from redshift z_start to z_end."""
        return None, None # spectrum, weight_box
    

class DMDecayInjection (Injection):
    """Dark matter decay injection object.
    
    Args:
        primary (str): Primary injection channel. See darkhistory.pppc.get_pppc_spec
        m_DM (float): DM mass in [eV].
        lifetime (float, optional): Decay lifetime in [s].
    """

    def __init__(self, primary, m_DM, lifetime):
        self.primary = primary
        self.m_DM = m_DM
        self.lifetime = lifetime

    def inj_phot_spec_box(self, z_start, z_end, **kwargs):
        return None, None # spectrum, weight_box

    def inj_elec_spec_box(self, z_start, z_end, **kwargs):
        return None, None # spectrum, weight_box