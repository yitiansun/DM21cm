import numpy as np
from scipy import special, optimize, interpolate


def decay_phot_lifetime(m):
    p = np.array([-0.38111888, 29.96460369])
    log10tau = np.polyval(p, np.log10(m))
    return 10 ** log10tau

def decay_elec_lifetime(m):
    p = np.array([  0.04765605,  -1.50826951,  15.03947904, -19.12576774])
    log10tau = np.polyval(p, np.log10(m))
    return 10 ** log10tau

def pwave_phot_c_sigma(m):
    log10m = np.log10(m)
    log10c = (log10m-9) * 1.3 - 15
    return 10 ** log10c

def pwave_elec_c_sigma(m):
    log10m = np.log10(m)
    log10c = (log10m-9) * 1.5 - 16
    return 10 ** log10c


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

def pbh_f(m):
    log10m = np.log10(m)
    l_func = lambda x: - 8 * (x - 14) - 16
    r_func = lambda x: 3.5 * (x - 14) - 14
    log10f = interp_between(l_func, r_func, 13.6, 14.2, log10m)
    return 10 ** log10f