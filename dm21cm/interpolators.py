"""Interpolators for energy depositions and secondary spectra."""

import pickle

import jax.numpy as jnp
from jax import jit, vmap
from functools import partial


##############################
## interpolating functions

def interp1d(f, x, xv):
    """Interpolates f(x) at values in xvs. Does not do bound checks.
    f : (n>=1 D) array of function value.
    x : 1D array of input value, corresponding to first dimension of f.
    xv : x values to interpolate.
    """
    li = jnp.searchsorted(x, xv) - 1
    lx = x[li]
    rx = x[li+1]
    p = (xv-lx) / (rx-lx)
    fl = f[li]
    return fl + (f[li+1]-fl) * p

#interp1d_vmap = vmap(interp1d, in_axes=(None, None, 0))


def interp2d(f, x0, x1, xv):
    """Interpolates f(x) at values in xvs. Does not do bound checks.
    f : (n>=2 D) array of function value.
    x0 : 1D array of input value, corresponding to first dimension of f.
    x1 : 1D array of input value, corresponding to second dimension of f.
    xv : [x0, x1] values to interpolate.
    """
    xv0, xv1 = xv
    
    li0 = jnp.searchsorted(x0, xv0) - 1
    lx0 = x0[li0]
    rx0 = x0[li0+1]
    p0 = (xv0-lx0) / (rx0-lx0)
    
    li1 = jnp.searchsorted(x1, xv1) - 1
    lx1 = x1[li1]
    rx1 = x1[li1+1]
    p1 = (xv1-lx1) / (rx1-lx1)
    
    fll = f[li0,li1]
    return fll + (f[li0+1,li1]-fll)*p0 + (f[li0,li1+1]-fll)*p1

interp2d_vmap = jit(vmap(interp2d, in_axes=(None, None, None, 0)))


##############################
## bound checking
def v_is_within(v, absc):
    """v can be value or vector."""
    return jnp.all(v >= jnp.min(absc)) and jnp.all(v <= jnp.max(absc))


##############################
## interpolator classes

class BatchInterpolator:
    """Interpolator for (rs, Ein, nBs, x, out) data. Vectorized in nBs and x
    directions.
    
    abscs : abscissas containing 'rs', 'Ein', 'nBs', 'x', and an out channel.
    data : grid data like (rs, Ein, nBs, x, out).
    
    Initialize with tuple (abscs, data) or a pickle filename containing this
    structure.
    """
    
    def __init__(self, abscs_data):
        
        if isinstance(abscs_data, str):
            abscs_data = pickle.load(open(abscs_data, 'rb'))
        self.abscs, self.data = abscs_data
    
    
    #@partial(jit, static_argnums=(0,))
    def __call__(self, rs, spec, nBs_s, x_s, out_of_bounds_action='error'):
        """Batch interpolate in nBs and x directions.
        
        Will first interpolate at a rs point, and sum with spec (a fixed
        spectral shape) in the second direction.
        """
        if out_of_bounds_action == 'clip':
            rs    = jnp.clip(rs   , jnp.min(self.abscs['rs']) , jnp.max(self.abscs['rs']))
            nBs_s = jnp.clip(nBs_s, jnp.min(self.abscs['nBs']), jnp.max(self.abscs['nBs']))
            x_s   = jnp.clip(x_s  , jnp.min(self.abscs['x'])  , jnp.max(self.abscs['x']))
        else:
            if not v_is_within(rs, self.abscs['rs']):
                raise ValueError('rs out of bounds.')
            if not v_is_within(nBs_s, self.abscs['nBs']):
                raise ValueError('nBs_s out of bounds.')
            if not v_is_within(x_s, self.abscs['x']):
                raise ValueError('x_s out of bounds.')
        
        data_at_rs = interp1d(self.data, self.abscs['rs'], rs)
        data_to_interp = jnp.einsum('i,ijkl->jkl', spec, data_at_rs)
        
        nBs_x_in = jnp.stack([nBs_s, x_s], axis=-1)
        return interp2d_vmap(data_to_interp,
                             self.abscs['nBs'], self.abscs['x'], nBs_x_in)