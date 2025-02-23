"""Abstract base class for injections."""

class Injection:
    """Abstract base class for injections.
    
    Methods:
        __init__          : Initialization.
        set_binning       : Set injection spectra according to binning chosen in dm21cm.evolve.
        is_injecting_elec : Whether DM is injecting electron/positron.
        get_config        : Get configuration of the injection. Used for reusability checks of cached solutions.

        inj_rate          : Injection event rate density.
        inj_power         : Injection power density.
        inj_phot_spec     : Injected photon rate density spectrum (in a homogeneous universe).
        inj_elec_spec     : Injected electron rate density spectrum (in a homogeneous universe).
        inj_phot_spec_box : Injected photon rate density spectrum and weight box.
        inj_elec_spec_box : Injected electron rate density spectrum and weight box.
    """

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
        """Whether DM is injecting electron/positron. Used by evolve.
        
        Returns:
            bool: Whether DM is injecting electron/positron.
        """
        raise NotImplementedError
    
    def get_config(self):
        """Get configuration of the injection.
        Used in DM21cm's DarkHistory wrapper to check if cached solution has the correct injection.
        
        Returns:
            dict: Configuration of the injection.
        """
        raise NotImplementedError

    def __eq__(self, other):
        """Equality comparison using self.get_config."""
        return self.get_config() == other.get_config()
    
    def __repr__(self):
        """Representation of the injection."""
        return f"{self.__class__.__name__}({self.get_config()})"

    #===== injections =====
    def inj_rate(self, z_start, z_end=None, state=None):
        """Injection event rate density in [inj / pcm^3 s].
        Used in DarkHistory. Assumes a homogeneous universe.
        If injection cannot be thought of as events, use 1 injection per second by convention.
        This factor is kept for DarkHistory's API, but will cancel out in the final result.

        Args:
            z_start (float): Starting redshift of the redshift step of injection.
            z_end (float, optional): Ending redshift of the redshift step of injection.
                Useful for calculating the average rate over a redshift step when injection rate evolves quickly.
                If None, must return instantaneous rate. This is used in DarkHistory's TLA integrator.
                For slow varying rates, it is sufficient to return the rate at the starting redshift for Euler steppping.
            state (dict, optional): State of the universe at z_start. Used for rates with feedback.

        Returns:
            float: Injection event rate per average baryon in [inj / Bavg s].
        """
        raise NotImplementedError
    
    def inj_power(self, z_start, z_end=None, state=None):
        """Injection power density in [eV / pcm^3 s].
        Used in DarkHistory. Assumes a homogeneous universe.

        Args:
            z_start (float): Starting redshift of the redshift step of injection.
            z_end (float, optional): Ending redshift of the redshift step of injection. See details in inj_rate.
            state (dict, optional): State of the universe at z_start. Used for rates with feedback.

        Returns:
            float: Injection power per average baryon in [eV / pcm^3 s].
        """
        raise NotImplementedError
    
    def inj_phot_spec(self, z_start, z_end=None, state=None, **kwargs):
        """Injected photon rate density spectrum assuming a homogeneous universe.
        Used in DarkHistory.

        Args:
            z_start (float): Starting redshift of the redshift step of injection.
            z_end (float, optional): Ending redshift of the redshift step of injection. See details in inj_rate.
            state (dict, optional): State of the universe at z_start. Used for rates with feedback.

        Returns:
            Spectrum: Injected photon rate spectrum in [phot / pcm^3 s].
                Spectrum value 'spec' can be either 'N' (particle in bin) or 'dNdE'.
                See darkhistory.spec.spectrum.Spectrum
        """
        raise NotImplementedError
    
    def inj_elec_spec(self, z_start, z_end=None, state=None, **kwargs):
        """Injected electron rate density spectrum assuming a homogeneous universe.
        Used in DarkHistory.

        Args:
            z_start (float): Starting redshift of the redshift step of injection.
            z_end (float, optional): Ending redshift of the redshift step of injection. See details in inj_rate.
            state (dict, optional): State of the universe at z_start. Used for rates with feedback.

        Returns:
            Spectrum: Injected electron rate spectrum in [spec / pcm^3 s].
                Spectrum value 'spec' can be either 'N' (particle in bin) or 'dNdE'.
                See darkhistory.spec.spectrum.Spectrum
        """
        raise NotImplementedError
    
    def inj_phot_spec_box(self, z_start, z_end=None, state=None, **kwargs):
        """Injected photon rate density spectrum and weight box.
        Called in dm21cm.evolve every redshift step.

        Args:
            z_start (float): Starting redshift of the redshift step of injection.
            z_end (float, optional): Ending redshift of the redshift step of injection. See details in inj_rate.
            state (dict, optional): State of the universe at z_start. Used for rates with feedback.

        Returns:
            tuple : (spec, weight_box), where:
                spec (Spectrum) : Injected photon rate density spectrum [spec / pcm^3 s].
                weight_box (ndarray) : Injection weight box of the above spectrum [1].

        Note:
            The output injection is spec \otimes weight_box, with spec carrying the units.
        """
        raise NotImplementedError

    def inj_elec_spec_box(self, z_start, z_end=None, state=None, **kwargs):
        """Injected electron rate density spectrum and weight box.
        Called in dm21cm.evolve every redshift step.

        Args:
            z_start (float): Starting redshift of the redshift step of injection.
            z_end (float, optional): Ending redshift of the redshift step of injection. See details in inj_rate.
            state (dict, optional): State of the universe at z_start. Used for rates with feedback.

        Returns:
            tuple : (spec, weight_box), where:
                spec (Spectrum) : Injected electron rate density spectrum [spec / pcm^3 s].
                weight_box (ndarray) : Injection weight box of the above spectrum [1].

        Note:
            The output injection is spec \otimes weight_box, with spec carrying the units.
        """
        raise NotImplementedError