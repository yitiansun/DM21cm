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

interp1d_vmap = jit(vmap(interp1d, in_axes=(None, None, 0)))


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
    """Interpolator for multidimensional data. Currently support
    axes = ('rs', 'Ein', 'nBs', 'x', 'out') and 
    axes = ('rs', 'Ein', 'x', 'out')
    
    abscs : abscissas of axes.
    axes : tuple of name of axes.
    data : grid data consistent with axes and abscs.
    
    Initialize with tuple (abscs, axes, data) or a pickle filename containing
    this structure.
    """
    
    def __init__(self, abscs_axes_data):
        
        if isinstance(abscs_axes_data, str):
            abscs_axes_data = pickle.load(open(abscs_axes_data, 'rb'))
        self.abscs, self.axes, self.data = abscs_axes_data
        self.fixed_spec = None
        self.fixed_spec_data = None
        
    
    def __call__(self, rs=None, in_spec=None, nBs_s=None, x_s=None,
                 return_sum=False, sum_batch_size=1000,
                 out_of_bounds_action='error'):
        """Batch interpolate in (nBs and) x directions.
        
        Will first sum with spec (with caching), then interpolate to a rs point,
        then perform the interpolation.
        
        return_sum : if True, return sum in the batch dimension.
        sum_batch_size : perform batch interpolation (and sum) in batches of this size.
        """
        
        if out_of_bounds_action == 'clip':
            rs    = jnp.clip(rs   , jnp.min(self.abscs['rs']) , jnp.max(self.abscs['rs']))
            if 'nBs' in self.axes:
                nBs_s = jnp.clip(nBs_s, jnp.min(self.abscs['nBs']), jnp.max(self.abscs['nBs']))
            x_s   = jnp.clip(x_s  , jnp.min(self.abscs['x'])  , jnp.max(self.abscs['x']))
        else:
            if not v_is_within(rs, self.abscs['rs']):
                raise ValueError('rs out of bounds.')
            if 'nBs' in self.axes:
                if not v_is_within(nBs_s, self.abscs['nBs']):
                    raise ValueError('nBs_s out of bounds.')
            if not v_is_within(x_s, self.abscs['x']):
                raise ValueError('x_s out of bounds.')
                
        if jnp.all(spec == 0):
            return jnp.zeros((len(x_s), len(self.abscs['out'])))
                
        # check if cached self.fixed_spec_data can be used
        if not jnp.all(spec == self.fixed_spec):
            if 'nBs' in self.axes:
                self.fixed_spec_data = jnp.einsum('e,renxo->rnxo', spec, self.data)
            else:
                self.fixed_spec_data = jnp.einsum('e,rexo->rxo', spec, self.data)
        
        data_at_rs = interp1d(self.fixed_spec_data, self.abscs['rs'], rs)
        
        if not return_sum: # interpolate only
            
            if 'nBs' in self.axes:
                nBs_x_in = jnp.stack([nBs_s, x_s], axis=-1)
                return interp2d_vmap(data_at_rs, self.abscs['nBs'], self.abscs['x'], nBs_x_in)
            else:
                return interp1d_vmap(data_at_rs, self.abscs['x'], x_s)
            
        else: # interpolate and sum
            
            split_n = int(jnp.ceil( len(x_s)/sum_batch_size ))
            result = np.zeros( (len(self.abscs['out']),) )
            
            if 'nBs' in self.axes:
                nBs_x_in = jnp.stack([nBs_s, x_s], axis=-1)
                nBs_x_in_batches = jnp.array_split(nBs_x_in, split_n)
                
                for nBs_x_in_batch in nBs_x_in_batches:
                    interp_result = interp2d_vmap(data_at_rs, self.abscs['nBs'], self.abscs['x'], nBs_x_in_batch)
                    result += np.array(jnp.sum(interp_result, axis=0))
            
            else:
                x_in_batches = jnp.array_split(x_s, split_n)
                
                for x_in_batch in x_in_batches:
                    interp_result = interp1d_vmap(data_at_rs, self.abscs['x'], x_in_batch)
                    result += np.array(jnp.sum(interp_result, axis=0))
                    
            return result
            