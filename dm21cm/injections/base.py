"""Abstract base class for injections."""

class Injection:

    def __init__(self):
        pass

    def set_binning(self, abscs):
        """Set injection spectra according to binning chosen in evolve.
        Called by evolve during initialization.

        Args:
            abscs (dict): Abscissas/binning for the run.
        """
        raise NotImplementedError

    def is_injecting_elec(self):
        """Whether DM is injecting electron/positron. Used by evolve."""
        raise NotImplementedError
    
    def get_config(self):
        """Get configuration of the injection.
        Used in DM21cm's DarkHistory wrapper to check if cached solution has the correct injection."""
        raise NotImplementedError

    def __eq__(self, other):
        """Equality comparison using self.get_config."""
        return self.get_config() == other.get_config()

    #===== injections =====
    def inj_rate_per_Bavg(self, z):
        """Injection event rate per average baryon in [inj / Bavg s].
        Used in DarkHistory. Assumes a homogeneous universe.
        If injection cannot be thought of as events, use any number, as this factor will be canceled.
        This factor is kept for DarkHistory's API.

        Args:
            z (float): (Starting) redshift of the redshift step of injection.

        Returns:
            float: Injection event rate per average baryon in [inj / Bavg s].
        """
        raise NotImplementedError
    
    def inj_power_per_Bavg(self, z):
        """Injection power per average baryon in [eV / Bavg s].
        Used in DarkHistory. Assumes a homogeneous universe.
        Different from `inj_rate_per_Bavg`, this factor affects DarkHistory run.

        Args:
            z (float): (Starting) redshift of the redshift step of injection.

        Returns:
            float: Injection power per average baryon in [eV / Bavg s].
        """
        raise NotImplementedError
    
    def inj_phot_spec(self, z, **kwargs):
        """Injected photon rate spectrum assuming a homogeneous universe.
        Used in DarkHistory.

        Args:
            z (float): (Starting) redshift of the redshift step of injection.

        Returns:
            Spectrum: Injected photon rate spectrum in [ph / Bavg s].
        """
        raise NotImplementedError
    
    def inj_elec_spec(self, z, **kwargs):
        """Injected electron rate spectrum assuming a homogeneous universe.
        Used in DarkHistory.

        Args:
            z (float): (Starting) redshift of the redshift step of injection.

        Returns:
            Spectrum: Injected photon rate spectrum in [e / Bavg s].
        """
        raise NotImplementedError
    
    def inj_phot_spec_box(self, z, **kwargs):
        """Injected photon rate spectrum and weight box.
        Called in dm21cm.evolve every redshift step.

        Args:
            z (float): (Starting) redshift of the redshift step of injection.

        Returns:
            (spec, weight_box) tuple, where:
                spec (Spectrum) : Injected photon rate spectrum [ph / Bavg s].
                weight_box (ndarray) : Injection weight box of the above spectrum [1].

        Note:
            The output injection is spec \otimes weight_box, with spec carrying the units.
        """
        raise NotImplementedError

    def inj_elec_spec_box(self, z, **kwargs):
        """Injected electron rate spectrum and weight box.
        Called in dm21cm.evolve every redshift step.

        Args:
            z (float): (Starting) redshift of the redshift step of injection.

        Returns:
            (spec, weight_box) tuple, where:
                spec (Spectrum) : Injected electron rate spectrum [e / Bavg s].
                weight_box (ndarray) : Injection weight box of the above spectrum [1].

        Note:
            The output injection is spec \otimes weight_box, with spec carrying the units.
        """
        raise NotImplementedError