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
        Dictionary of abscissa. Should have 'in_eng' and 'out_eng'.


    Attributes
    ----------
    grid_vals : ndarray
    abscs : dict
    """

    def __init__(self, grid_vals, abscs):
        self.grid_vals = grid_vals
        self.abscs = abscs # in_eng, out_eng
        
    @property
    def extent(self):
        return (self.abscs['out_eng'][0], self.abscs['out_eng'][-1],
                self.abscs['in_eng'][0],  self.abscs['in_eng'][-1])
    
    def __call__(self, spec):
        if isinstance(spec, Spectrum):
            out_spec_N = np.dot(spec.N, self.grid_vals)
        elif isinstance(spec, ndarray):
            out_spec_N = np.dot(spec, self.grid_vals)
        else:
            raise ValueError("spec is not of type Spectrum or ndarray.")
        return Spectrum(self.abscs['out_eng'], out_spec_N, spec_type='N')


class Depositions:
    """Simple class for calculating energy deposition for injected photons.
    
    Parameters
    ----------
    grid_vals : ndarray
        Grid values of transfer function. Row index: ingoing energy; column
        index: channel.
    abscs : dict
        Dictionary of abscissa. Should have 'in_eng' and 'channel'.


    Attributes
    ----------
    grid_vals : ndarray
    abscs : dict
    """
    
    def __init__(self, grid_vals, abscs):
        self.grid_vals = grid_vals
        self.abscs = abscs # in_eng, channel
            
    def __call__(self, spec):
        deps = np.dot(spec.N, self.grid_vals)
        return { c: deps[i] for i, c in enumerate(self.abscs['channel']) }
        
        
EPSILON = 1e-100
        
class Interpolator:
    """Interpolator.
    
    Parameters
    ----------
    grid_vals : ndarray
        Grid point values of the array of transfer functions (including in_eng
        and out_eng).
    abscs : dict
        Dict for the transfer function abscissas (including in_eng and out_eng).
    interp_axes : list
        List of abscissa axis names interpolated (e.g. ['nBs', 'x', 'rs']).
    log_interp : bool
        If True, performs an interpolation of log of the grid values over log of
        abscissa values.
        
    Attributes
    ----------
    grid_vals : ndarray
    abscs : dict
    interp_axes : list
    log_interp : bool
    
    in_func : function
    out_func : function
    interpolator : function
    """
    
    def __init__(self, grid_vals, abscs, interp_axes, log_interp=False):

        self.grid_vals = grid_vals
        self.abscs = abscs
        self.interp_axes = interp_axes
        self.log_interp = log_interp
        
        if self.log_interp:
            self.grid_vals[self.grid_vals <= 0] = EPSILON
            self.in_func = np.log
            self.out_func = np.exp
        else:
            self.in_func = lambda x: x
            self.out_func = lambda x: x
            
        interp_abscs = [self.in_func(abscs[name]) for name in self.interp_axes]
        self.interpolator = interpolate.RegularGridInterpolator(
            tuple(interp_abscs), self.in_func(self.grid_vals)
        )
        
    def __call__(self, **kwargs):
        
        coord = [self.in_func(kwargs[name]) for name in self.interp_axes]
        return self.out_func(np.squeeze(self.interpolator(coord)))
        
        
class TransferFunctionInterpolator(Interpolator):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
    def __call__(self, **kwargs):
        
        abscs = { k: self.abscs[k] for k in ['in_eng', 'out_eng'] }
        return TransferFunction(super().__call__(**kwargs), abscs)

    
class DepositionsInterpolator(Interpolator):
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
    def __call__(self, **kwargs):
        
        abscs = { k: self.abscs[k] for k in ['in_eng', 'channel'] }
        return Depositions(super().__call__(**kwargs), abscs)