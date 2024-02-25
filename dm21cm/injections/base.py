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
    def inj_phot_spec_box(self, z_start, dt, **kwargs):
        """Injected photon spectrum and weight box starting from redshift z_start, for duration dt.
        Called by DM21cm's transfer functions wrapper every redshift step.

        Args:
            z_start (float): Starting redshift of the redshift step of injection.
            dt (float): Duration of the redshift step in [s]. Specified in evolve to avoid inconsistent
                dt calculations.

        Returns:
            (spec, weight_box) tuple, where:
                spec (Spectrum) : Injected photon spectrum [ph / Bavg] (number of photons per average Baryon).
                weight_box (ndarray) : Injection weight box of the above spectrum [1].

        Note:
            The output injection is spec \otimes weight_box, with spec carrying the units.
        """
        raise NotImplementedError

    def inj_elec_spec_box(self, z_start, dt, **kwargs):
        """Injected electron spectrum and weight box starting from redshift z_start, for duration dt.
        Called by DM21cm's transfer functions wrapper every redshift step. See `inj_phot_spec_box` for details.
        """
        raise NotImplementedError

    def inj_phot_spec(self, z_start, dt, **kwargs):
        """Injected photon spectrum similar to `inj_phot_spec_box` assuming a homogeneous universe.
        Called by DarkHistory's evolve.

        Args:
            z_start (float): Starting redshift of the redshift step of injection.
            dt (float): Duration of the redshift step in [s]. Specified in evolve to avoid inconsistent
                dt calculations.

        Returns:
            Spectrum: Injected photon spectrum [ph / Bavg] (number of photons per average Baryon).
        """
        raise NotImplementedError
    
    def inj_elec_spec(self, z_start, dt, **kwargs):
        """Injected electron spectrum similar to `inj_elec_spec_box` assuming a homogeneous universe.
        Called by DarkHistory's evolve. See inj_phot_spec for details.
        """
        raise NotImplementedError

    def inj_E_per_Bavg(self):
        """Total energy injected in redshift step per average Baryon [eV/Bavg] in dt.
        Called by DM21cm.evolve for recording."""
        raise NotImplementedError