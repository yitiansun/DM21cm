import numpy as np
from scipy import integrate


#########################################
# Units in cgs                          #
#########################################

Mpc = 3.08568e24
"""Mpc in cm."""


#########################################
# Fundamental Particles and Constants   #
#########################################
# Based on darkhistory.physics

m_p          = 0.938272081e9
"""Proton mass in eV."""
m_e          = 510998.9461
"""Electron mass in eV."""
m_He         = 3.97107*m_p
"""Helium nucleus mass in eV."""
hbar         = 6.58211951e-16
"""hbar in eV s."""
c            = 299792458e2
"""Speed of light in cm s\ :sup:`-1`\ ."""
kB           = 8.6173324e-5
"""Boltzmann constant in eV K\ :sup:`-1`\ ."""

rydberg      = 13.60569253
"""Ionization potential of ground state hydrogen in eV."""
lya_eng      = rydberg*3/4
"""Lyman alpha transition energy in eV."""


#########################################
# Densities and Hubble                  #
#########################################
# Based on darkhistory.physics

h    = 0.6736
""" h parameter."""
H0   = 100*h*3.241e-20
""" Hubble parameter today in s\ :sup:`-1`\ ."""

omega_m      = 0.3153
""" Omega of all matter today."""
omega_rad    = 8e-5
""" Omega of radiation today."""
omega_lambda = 0.6847
""" Omega of dark energy today."""
omega_baryon = 0.02237/(h**2)
""" Omega of baryons today."""
omega_DM     = 0.1200/(h**2)
""" Omega of dark matter today."""
rho_crit     = 1.05371e4*(h**2)
""" Critical density of the universe in eV cm\ :sup:`-3`\ ."""
rho_DM       = rho_crit*omega_DM
""" DM density in eV cm\ :sup:`-3`\ ."""
rho_baryon   = rho_crit*omega_baryon
""" Baryon density in eV cm\ :sup:`-3`\ ."""
n_B          = rho_baryon/m_p
""" Baryon number density in cm\ :sup:`-3`\ ."""

Y_He         = 0.245
"""Helium abundance by mass."""
n_H          = (1-Y_He) * n_B
""" Atomic hydrogen number density in cm\ :sup:`-3`\ ."""
n_He         = (Y_He/4) * n_B
""" Atomic helium number density in cm\ :sup:`-3`\ ."""
n_A          = n_H + n_He
""" Hydrogen and helium number density in cm\ :sup:`-3`\ .""" 
chi          = n_He / n_H
"""Ratio of helium to hydrogen nuclei."""


def hubble(rs, H0=H0, omega_m=omega_m, omega_rad=omega_rad, omega_lambda=omega_lambda):
    """ Hubble parameter in s\ :sup:`-1`\ .
    (Copied from darkhistory.physics.hubble)

    Assumes a flat universe.

    Parameters
    ----------
    rs : float
        The redshift of interest (rs = 1+z).
    H0 : float
        The Hubble parameter today, default value `H0`.
    omega_m : float, optional
        Omega matter today, default value `omega_m`.
    omega_rad : float, optional
        Omega radiation today, default value `omega_rad`.
    omega_lambda : float, optional
        Omega dark energy today, default value `omega_lambda`.

    Returns
    -------
    float
    """
    
    return H0*np.sqrt(omega_rad*rs**4 + omega_m*rs**3 + omega_lambda)


def dtdz(rs, H0=H0, omega_m=omega_m, omega_rad=omega_rad, omega_lambda=omega_lambda):
    """ dt/dz in s.
    (Based on darkhistory.physics.inj_rate)

    Parameters
    ----------
    rs : float
        The redshift of interest (rs = 1+z).
    H0 : float
        The Hubble parameter today, default value `H0`.
    omega_m : float, optional
        Omega matter today, default value `omega_m`.
    omega_rad : float, optional
        Omega radiation today, default value `omega_rad`.
    omega_lambda : float, optional
        Omega dark energy today, default value `omega_lambda`.

    Returns
    -------
    float
    """

    return -1./(rs*hubble(rs, H0, omega_m, omega_rad, omega_lambda))


def dt_between_z(z_high, z_low, **kwargs):
    """ Calculate delta t [s] between z_high and z_low.
    
    Parameters
    ----------
    z_high: float
    z_low: float
    kwargs: dict
        Takes H0, omega_m, omega_rad, omega_lambda. See dtdz.
    
    Returns
    -------
    float
    """
    
    dtdz_zhigh = dtdz(1+z_high, **kwargs)
    integrand = lambda z: dtdz(1+z, **kwargs) / dtdz_zhigh
    dt, dt_err = np.array(integrate.quad(integrand, z_high, z_low)) * dtdz_zhigh
    
    if dt_err / dt > 1e-6:
        raise ValueError('large integral error.')
    
    return dt


#########################################
# Dark Matter                           #
#########################################

def inj_rate_box(rho_DM_box, mode=None, mDM=None, sigmav=None, lifetime=None):
    """ Dark matter annihilation/decay energy injection rate box.
    (Based on darkhistory.physics.inj_rate)

    Parameters
    ----------
    rho_DM_box : ndarray (3D)
        DM density box at redshift in eV cm\ :sup:`-3`\ .
    mode : {'swave', 'decay'}
        Type of injection.
    mDM : float, optional
        DM mass in eV.
    sigmav : float, optional
        Annihilation cross section in cm\ :sup:`3`\ s\ :sup:`-1`\ .
    lifetime : float, optional
        Decay lifetime in s.

    Returns
    -------
    ndarray
        The dE/dV_dt injection rate box in eV cm\ :sup:`-3`\ s\ :sup:`-1`\ .

    """
    if mode == 'swave':
        return rho_DM_box**2 * sigmav / mDM
    
    elif mode == 'decay':
        return rho_DM_box / lifetime
    
    else:
        raise ValueError('Unknown mode.')
        
        
def struct_boost_func(model='einasto_subs', model_params=None):
    """Structure formation boost factor 1+B(z).
    (Copied from darkhistory.physics.struct_boost_func)

    Parameters
    ----------
    model : {'einasto_subs', 'einasto_no_subs', 'NFW_subs', 'NFW_no_subs', 'erfc'}
        Model to use. See 1604.02457. 
    model_params : tuple of floats
        Model parameters (b_h, delta, z_h) for 'erfc' option. 

    Returns
    -------
    float or ndarray
        Boost factor. 

    Notes
    -----
    Refer to 1408.1109 for erfc model, 1604.02457 for all other model
    descriptions and parameters.

    """

    if model == 'erfc':

        from scipy.special import erfc

        if model_params is None:
            # Smallest boost in 1408.1109. 
            b_h   = 1.6e5
            delta = 1.54
            z_h   = 19.5
        else:
            b_h   = model_params[0]
            delta = model_params[1]
            z_h   = model_params[2]

        def func(rs):
            return 1. + b_h / rs**delta * erfc( rs/(1+z_h) )

    else:

        struct_data = load_data('struct')[model]
        log_struct_interp = interp1d(
            np.log(struct_data[:,0]), np.log(struct_data[:,1]),
            bounds_error=False, fill_value=(np.nan, 0.)
        )

        def func(rs):
            return np.exp(log_struct_interp(np.log(rs)))

    return func