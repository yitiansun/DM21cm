"""Interpolators for energy depositions and secondary spectra."""

import h5py

import numpy as np

import jax.numpy as jnp
from jax import jit, vmap
from functools import partial


#===== interpolating functions =====

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


#===== utilities =====

def v_is_within(v, absc):
    """v can be value or vector."""
    return jnp.all(v >= jnp.min(absc)) and jnp.all(v <= jnp.max(absc))


#===== interpolator class =====

class BatchInterpolator:
    """Interpolator for multidimensional data. Currently support
    axes = ('rs', 'Ein', 'nBs', 'x', 'out')
    
    Parameters
    ----------
    filename : str
        HDF5 data file name.
        
    Attributes
    ---------
    axes : list
        List of axes.
    abscs : dict
        Abscissas of axes.
    data : array
        Grid data consistent with axes and abscs.
    """
    
    def __init__(self, filename):
        
        with h5py.File(filename, 'r') as hf:
            self.axes = hf['axes'][:]
            self.abscs = {}
            for k, item in hf['abscs'].items():
                self.abscs[k] = item[:]
            self.data = hf['data'][:] # load into memory
        
        self.fixed_in_spec = None
        self.fixed_in_spec_data = None
        
        
    def set_fixed_in_spec(self, in_spec):
        
        self.fixed_in_spec = in_spec
        self.fixed_in_spec_data = jnp.einsum('e,renxo->rnxo', in_spec, self.data)
        
        
    def __call__(self, rs=None, in_spec=None, nBs_s=None, x_s=None,
                 sum_result=False, sum_weight=None, sum_batch_size=10000,
                 out_of_bounds_action='error'):
        """Batch interpolate in (nBs and) x directions.
        
        First sum with in_spec (with caching), then interpolate to a rs point,
        then perform the interpolation on [(nBs_s), x_s]. If sum_result is True,
        sum over all interpolated value.
        
        Parameters:
            rs : [1]
            in_spec : [N * ...]
            nBs_s : [1]
            x_s : [1]
            sum_result : if True, return average in the batch dimension.
            sum_weight : if None, just sum. otherwise dot.
            sum_batch_size : perform batch interpolation (and averaging) in batches of this size.
            out_of_bounds_action : {'error', 'clip'}
        
        Return:
            interpolated box or average
        """
        
        if out_of_bounds_action == 'clip':
            rs  = jnp.clip(rs,  jnp.min(self.abscs['rs']), jnp.max(self.abscs['rs']))
            x_s = jnp.clip(x_s, jnp.min(self.abscs['x']),  jnp.max(self.abscs['x']))
            nBs_s = jnp.clip(nBs_s, jnp.min(self.abscs['nBs']), jnp.max(self.abscs['nBs']))
        else:
            if not v_is_within(rs, self.abscs['rs']):
                raise ValueError('rs out of bounds.')
            if not v_is_within(x_s, self.abscs['x']):
                raise ValueError('x_s out of bounds.')
            if not v_is_within(nBs_s, self.abscs['nBs']):
                raise ValueError('nBs_s out of bounds.')
        
        ## 1. in_spec sum
        if jnp.all(in_spec == self.fixed_in_spec):
            in_spec_data = self.fixed_in_spec_data
        else:
            in_spec_data = jnp.einsum('e,renxo->rnxo', in_spec, self.data)
        
        ## 2. rs interpolation
        data_at_rs = interp1d(in_spec_data, self.abscs['rs'], rs)
        
        if not sum_result:
            
            ## 3. (nBs) x interpolation
            nBs_x_in = jnp.stack([nBs_s, x_s], axis=-1)
            return interp2d_vmap(
                data_at_rs,
                self.abscs['nBs'],
                self.abscs['x'],
                nBs_x_in
            )
            
        else:
            ## 3. (nBs) x sum
            split_n = int(jnp.ceil( len(x_s)/sum_batch_size ))
            if sum_weight is not None:
                sum_weight_batches = jnp.array_split(sum_weight, split_n)
                
            result = jnp.zeros( (len(self.abscs['out']),) ) # use numpy?
            
            nBs_x_in = jnp.stack([nBs_s, x_s], axis=-1)
            nBs_x_in_batches = jnp.array_split(nBs_x_in, split_n)

            for i_batch, nBs_x_in_batch in enumerate(nBs_x_in_batches):
                interp_result = interp2d_vmap(
                    data_at_rs,
                    self.abscs['nBs'],
                    self.abscs['x'],
                    nBs_x_in_batch
                )
                if sum_weight is None:
                    result += jnp.sum(interp_result, axis=0)
                else:
                    result += jnp.dot(sum_weight_batches[i_batch], interp_result)
                    
            return result
        
    
    def point_interp(self, rs=None, nBs=None, x=None):
        """Returns the transfer function at a (rs, nBs, x) point."""
        
        data = interp1d(self.data, self.abscs['rs'], rs) # enxo
        data = np.einsum('enxo -> nxeo', data) # nxeo
        data = interp1d(data, self.abscs['nBs'], nBs) # xeo
        data = interp1d(data, self.abscs['x'], x) # eo
        return data