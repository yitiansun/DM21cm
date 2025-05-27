import numpy as np
from scipy import special, optimize, interpolate


#===== helper functions =====

def bernstein_poly(i, n, t):
    """The Bernstein polynomial of n, i as a function of t"""
    return special.comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, x_in):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, 100)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(nPoints) ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return interpolate.interp1d(xvals, yvals, fill_value='extrapolate')(x_in)

def interp_between(l_func, r_func, l_bound, r_bound, x):
    """Interpolate between two functions."""
    p_l = [l_bound, l_func(l_bound)]
    p_r = [r_bound, r_func(r_bound)]
    x_intersect = optimize.brentq(lambda x: l_func(x) - r_func(x), l_bound, r_bound)
    p_m = [x_intersect, l_func(x_intersect)]

    return np.where(x < l_bound,
        l_func(x),
        np.where(x > r_bound,
            r_func(x),
            bezier_curve([p_l, p_m, p_r], x)
        )
    )


#===== step sizes =====

decay_phot_m_s = np.logspace(1.5, 12, 22) # runs for 2312.11608
decay_elec_m_s = np.logspace(6.5, 12, 23) # runs for 2312.11608

def decay_phot_lifetime(m):
    """Decay to photons lifetime step size [s] for a given DM mass [eV]."""
    p = np.array([-0.38111888, 29.96460369])
    log10tau = np.polyval(p, np.log10(m))
    return 10 ** log10tau

def decay_elec_lifetime(m):
    """Decay to electrons lifetime step size [s] for a given DM mass [eV]."""
    p = np.array([  0.04765605,  -1.50826951,  15.03947904, -19.12576774])
    log10tau = np.polyval(p, np.log10(m))
    return 10 ** log10tau

pwave_phot_m_s = np.logspace(1.5, 12, 22)
pwave_elec_m_s = np.logspace(6.5, 12, 12)

def pwave_phot_c_sigma(m):
    """P-wave annihilation to photons cross section at v=c step size [pcm^3/s] for a given DM mass [eV]."""
    log10m = np.log10(m)
    log10c = (log10m-12) * 1.6 - 12.1
    return 10 ** log10c

# def pwave_elec_c_sigma(m):
# """P-wave annihilation to electrons cross section at v=c step size [pcm^3/s] for a given DM mass [eV]."""
#     log10m = np.log10(m)
#     log10c = (log10m-9) * 1.5 - 19
#     return 10 ** log10c

def pwave_elec_c_sigma(m):
    """P-wave annihilation to electrons cross section at v=c step size [pcm^3/s] for a given DM mass [eV]."""
    log10m = np.log10(m)
    log10c = bezier_curve([[6.5, -23], [10.5, -20], [12, -14.5]], log10m)
    return 10 ** log10c

def pwave_tau_c_sigma(m):
    """P-wave annihilation to tautau cross section at v=c step size [pcm^3/s] for a given DM mass [eV]."""
    log10m = np.log10(m)
    log10c = (log10m-12) * 1.8 - 14.8
    return 10 ** log10c

# def pwave_tau_c_sigma_old(m):
#     """P-wave annihilation to tautau cross section at v=c step size [pcm^3/s] for a given DM mass [eV]."""
#     log10m = np.log10(m)
#     log10c = (log10m-12) * 1.6 - 12.1
#     return 10 ** log10c

pbh_hr_m_s = None
pbh_acc_m_s = None

def pbhhr_f(m):
    """PBH fraction step size [1] for a given PBH mass [g]."""
    log10m = np.log10(m)
    l_func = lambda x: - 8 * (x - 14) - 16
    r_func = lambda x: 3.5 * (x - 14) - 14
    log10f = interp_between(l_func, r_func, 13.6, 14.2, log10m)
    return 10 ** log10f

def pbhacc_f(m, model):
    """PBH accretion fraction step size [1] for a given PBH mass [Msun]."""
    log10m = np.log10(m)
    if model in ['PRc23', 'PRc23R', 'PRc23H']:
        log10f = -1.5 * log10m - 0.75
    elif model in ['PRc50']:
        log10f = -1.5 * log10m + 0
    elif model in ['PRc10']:
        log10f = -1.4 * log10m - 3.5
    elif model == 'BHLl2':
        log10f = -1.0 * log10m - 8
    else:
        raise NotImplementedError(model)
    return 10 ** log10f

def pbhacc_f_old(m, model):
    """PBH accretion fraction step size [1] for a given PBH mass [Msun]."""
    log10m = np.log10(m)
    if model in ['PRc23', 'PRc23R', 'PRc23H']:
        log10f = -1.5 * log10m - 0.75
    elif model in ['PRc50']:
        log10f = -1.5 * log10m - 0.4
    elif model in ['PRc10']:
        log10f = -1.4 * log10m - 3.5
    elif model == 'BHLl2':
        log10f = -1.0 * log10m - 8
    else:
        raise NotImplementedError(model)
    return 10 ** log10f