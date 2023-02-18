"""Classes for interpolating transfer functions. See darkhistory.spec."""

import numpy as np
from scipy import interpolate

import os, sys
sys.path.append(os.environ['DH_DIR'])
from darkhistory.spec.spectrum import Spectrum


class TransferFunction:
    """Transfer function that maps Spectrum to Spectrum. (N -> N)

    Parameters
    ----------
    grid_vals : ndarray
        Grid values of transfer function. Row index: ingoing energy; column
        index: outgoing energy.
    abscs : dict
        Dictionary of abscissa.
    io_axes : list
        Names of input and output abscissa.


    Attributes
    ----------
    grid_vals : ndarray
    abscs : dict
    in_abscs : ndarray
    out_abscs : ndarray
    """

    def __init__(self, grid_vals, abscs, io_axes):
        self.grid_vals = grid_vals
        self.abscs = abscs
        self.in_abscs = abscs[io_axes[0]]
        self.out_abscs = abscs[io_axes[1]]
        
    @property
    def extent(self):
        return (self.out_abscs[0], self.out_abscs[-1],
                self.in_abscs[0],  self.in_abscs[-1])
    
    def __call__(self, spec):
        if isinstance(spec, Spectrum):
            out_spec_N = np.dot(spec.N, self.grid_vals)
        elif isinstance(spec, ndarray):
            out_spec_N = np.dot(spec, self.grid_vals)
        else:
            raise ValueError("spec is not of type Spectrum or ndarray.")
        return Spectrum(self.out_abscs, out_spec_N, spec_type='N')


class Depositions:
    """Simple class for calculating energy deposition for injected photons.
    
    Parameters
    ----------
    grid_vals : ndarray
        Grid values of transfer function. Row index: ingoing energy; column
        index: channel.
    abscs : dict
        Dictionary of abscissa.
    io_axes : list
        Names of input and output abscissa.


    Attributes
    ----------
    grid_vals : ndarray
    abscs : dict
    in_abscs : ndarray
    out_abscs : ndarray
    """
    
    def __init__(self, grid_vals, abscs, io_axes):
        self.grid_vals = grid_vals
        self.abscs = abscs
        self.in_abscs = abscs[io_axes[0]]
        self.out_abscs = abscs[io_axes[1]]
            
    def __call__(self, spec):
        deps = np.dot(spec.N, self.grid_vals)
        return { c: deps[i] for i, c in enumerate(self.out_abscs) }
        
        
EPSILON = 1e-100
        
class Interpolator:
    """Interpolator.
    
    Parameters
    ----------
    tf_class: class
        Class of the transfer function being interpolated.
    grid_vals : ndarray
        Grid point values of the array of transfer functions (including in_eng
        and out_eng).
    abscs : dict
        Dict for the transfer function abscissas (including in_eng and out_eng).
    interp_axes : list
        List of abscissa axis names interpolated (e.g. ['nBs', 'x', 'rs']).
    io_axes : list
        Names of input and output axes of transfer function being interpolated
        (e.g. ['photE', 'phoE']).
    log_interp : bool
        If True, performs an interpolation of log of the grid values over log of
        abscissa values.
        
    Attributes
    ----------
    tf_class : class
    grid_vals : ndarray
    abscs : dict
    interp_axes : list
    log_interp : bool
    interpolator : function
    """
    
    def __init__(self, tf_class, grid_vals, abscs, interp_axes, io_axes,
                 log_interp=False):

        self.tf_class = tf_class
        self.grid_vals = grid_vals
        self.abscs = abscs
        self.interp_axes = interp_axes
        self.io_axes = io_axes
        self.log_interp = log_interp
        
        if self.log_interp:
            self.grid_vals[self.grid_vals <= 0] = EPSILON
            in_func = np.log
            out_func = np.exp
        else:
            in_func = lambda x: x
            out_func = lambda x: x
            
        interp_abscs = [in_func(abscs[name]) for name in self.interp_axes]
        self.interpolator = interpolate.RegularGridInterpolator(
            tuple(interp_abscs), in_func(self.grid_vals)
        )
        
    def __call__(self, **kwargs):
        
        if self.log_interp:
            in_func = np.log
            out_func = np.exp
        else:
            in_func = lambda x: x
            out_func = lambda x: x
            
        coord = [in_func(kwargs[name]) for name in self.interp_axes]
        tf_grid_vals = out_func(np.squeeze(self.interpolator(coord)))
        return self.tf_class(tf_grid_vals, self.abscs, self.io_axes)